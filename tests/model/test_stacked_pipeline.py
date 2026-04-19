"""Regression tests for the end-to-end stacked model training pipeline."""

import numpy as np
import pytest

from src.pipeline.stacked_pipeline import StackedModelTrainingPipeline
from tests.helpers import make_ohlcv, fast_xgb_params, fast_lgb_params

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StubDownloader:
    """Minimal stand-in for StockDataDownloader that returns deterministic values."""

    def get_sector(self, symbol: str) -> str:
        return "Technology"

    def get_industry(self, symbol: str) -> str:
        return "Software"

    def get_beta(self, symbol: str) -> float:
        return 1.0

    def get_avg_volume(self, symbol: str) -> float:
        return 1_000_000.0

    def get_market_cap(self, symbol: str) -> float:
        return 1_000_000_000.0


def _build_pipeline(symbols, monkeypatch) -> StackedModelTrainingPipeline:
    """
    Construct a StackedModelTrainingPipeline pre-loaded with synthetic data
    so that download_data() and external I/O are never called.
    """
    intervals = ["1d", "1h"]

    # Use the shared fast params, but reduce n_estimators further for pipeline speed.
    pipeline_xgb = {**fast_xgb_params(), "n_estimators": 10}
    pipeline_lgb = {**fast_lgb_params(), "n_estimators": 10}

    pipeline = StackedModelTrainingPipeline(
        intervals=intervals,
        symbols=symbols,
        test_size=0.2,
    )
    pipeline._xgb_params = pipeline_xgb
    pipeline._lgb_params = pipeline_lgb

    # Inject synthetic OHLCV data (same for both intervals in tests).
    stock_data = {}
    for i, interval in enumerate(intervals):
        stock_data[interval] = {symbol: make_ohlcv(n=200, seed=j + i * 100) for j, symbol in enumerate(symbols)}
    pipeline._stock_data = stock_data
    pipeline._downloader = _StubDownloader()

    # Suppress file I/O that happens at the end of train().
    monkeypatch.setattr(
        "src.pipeline.stacked_pipeline.save_trained_model",
        lambda *args, **kwargs: None,
    )

    return pipeline


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def prepared_pipeline(monkeypatch):
    """Pipeline with prepare_features() already called."""
    symbols = ["AAPL", "MSFT", "GOOGL"]
    pipeline = _build_pipeline(symbols, monkeypatch)
    pipeline.prepare_features()
    return pipeline


@pytest.fixture
def trained_pipeline(monkeypatch):
    """Pipeline that has been fully trained."""
    symbols = ["AAPL", "MSFT", "GOOGL"]
    pipeline = _build_pipeline(symbols, monkeypatch)
    pipeline.prepare_features()
    pipeline.train()
    return pipeline


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStackedPipelineRegression:
    """Regression tests using small deterministic synthetic data."""

    def test_pipeline_produces_predictions(self, trained_pipeline):
        """Ensure the pipeline runs end-to-end and returns predictions."""
        predictions = trained_pipeline._get_predictions()
        assert predictions is not None
        assert len(predictions) > 0

    def test_prediction_shape_matches_input(self, trained_pipeline):
        """Predictions array length should match the test set size."""
        predictions = trained_pipeline._get_predictions()
        y_test = trained_pipeline._y_test
        assert len(predictions) == len(y_test), (
            f"Prediction length {len(predictions)} != test target length {len(y_test)}"
        )

    def test_predictions_are_deterministic(self, monkeypatch):
        """Two pipeline runs with the same seed produce identical predictions."""
        symbols = ["AAPL", "MSFT"]

        p1 = _build_pipeline(symbols, monkeypatch)
        p1.prepare_features()
        p1.train()
        preds1 = p1._get_predictions()

        p2 = _build_pipeline(symbols, monkeypatch)
        p2.prepare_features()
        p2.train()
        preds2 = p2._get_predictions()

        np.testing.assert_array_almost_equal(
            preds1,
            preds2,
            decimal=5,
            err_msg="Pipeline predictions are not deterministic across identical runs",
        )

    def test_model_metrics_within_expected_range(self, trained_pipeline):
        """R², RMSE, MAE should be finite and within historically observed ranges."""
        results = trained_pipeline._test_results
        assert results is not None, "Test results must be populated after train()"

        r2 = results["r2"]
        rmse = results["rmse"]
        mae = results["mae"]

        assert np.isfinite(r2), f"R² is not finite: {r2}"
        assert np.isfinite(rmse), f"RMSE is not finite: {rmse}"
        assert np.isfinite(mae), f"MAE is not finite: {mae}"

        assert r2 > -2.0, f"R² is unexpectedly low: {r2:.4f}"
        assert rmse > 0.0, f"RMSE should be positive, got: {rmse}"
        assert mae > 0.0, f"MAE should be positive, got: {mae}"

    def test_pipeline_handles_missing_data_gracefully(self, monkeypatch):
        """Pipeline should skip tickers with insufficient data, not raise an exception."""
        symbols = ["VALID", "TINY"]

        pipeline = StackedModelTrainingPipeline(
            intervals=["1d", "1h"],
            symbols=symbols,
            test_size=0.2,
        )
        pipeline._xgb_params = {**fast_xgb_params(), "n_estimators": 10}
        pipeline._lgb_params = {**fast_lgb_params(), "n_estimators": 10}

        stock_data = {}
        for interval in ["1d", "1h"]:
            stock_data[interval] = {
                "VALID": make_ohlcv(n=200, seed=1),
                # Only 5 rows — too few to produce any valid features/targets.
                "TINY": make_ohlcv(n=5, seed=2),
            }
        pipeline._stock_data = stock_data
        pipeline._downloader = _StubDownloader()

        monkeypatch.setattr(
            "src.pipeline.stacked_pipeline.save_trained_model",
            lambda *args, **kwargs: None,
        )

        # Should not raise; TINY is silently skipped.
        pipeline.prepare_features()

        # VALID ticker must still have contributed data.
        default_iv = pipeline.default_interval
        assert len(pipeline._test_train_data["train"][default_iv]) > 0

    def test_prepare_features_raises_before_data_download(self):
        """prepare_features() should raise when called before _stock_data is set."""
        pipeline = StackedModelTrainingPipeline(
            intervals=["1d", "1h"],
            symbols=["AAPL"],
        )
        with pytest.raises(Exception, match="download_data"):
            pipeline.prepare_features()

    def test_train_raises_before_prepare_features(self):
        """train() should raise when called before prepare_features()."""
        pipeline = StackedModelTrainingPipeline(
            intervals=["1d", "1h"],
            symbols=["AAPL"],
        )
        with pytest.raises(Exception, match="prepare_features"):
            pipeline.train()

    def test_feature_partitions_populated_after_prepare(self, prepared_pipeline):
        """After prepare_features(), feature partition dicts must be filled."""
        assert prepared_pipeline.feature_partitions, "feature_partitions should not be empty"
        assert prepared_pipeline.interval_feature_sets, "interval_feature_sets should not be empty"

        for interval in prepared_pipeline.intervals:
            assert interval in prepared_pipeline.feature_partitions
            assert interval in prepared_pipeline.interval_feature_sets

    def test_test_train_data_has_expected_keys(self, prepared_pipeline):
        """_test_train_data must contain 'train' and 'test' splits with required keys."""
        ttd = prepared_pipeline._test_train_data
        assert "train" in ttd
        assert "test" in ttd

        for split in ("train", "test"):
            for key in ("targets", "dates", "prices", "symbols"):
                assert key in ttd[split], f"Key '{key}' missing from '{split}' split"

    def test_stock_data_freed_after_prepare_features(self, prepared_pipeline):
        """_stock_data must be None after prepare_features() to release raw OHLCV memory."""
        assert prepared_pipeline._stock_data is None, "_stock_data should be None after prepare_features()"

    def test_feature_matrix_uses_float32(self, prepared_pipeline):
        """Numeric feature columns in the train/test splits must be float32, not float64."""
        default_iv = prepared_pipeline.default_interval
        for split in ("train", "test"):
            df = prepared_pipeline._test_train_data[split][default_iv]
            float64_cols = [c for c in df.columns if df[c].dtype == "float64"]
            assert float64_cols == [], (
                f"Expected no float64 columns in {split} split, found: {float64_cols}"
            )
