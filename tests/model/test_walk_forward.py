"""Tests for walk-forward cross-validation — WalkForwardValidator and pipeline integration."""

import numpy as np
import pandas as pd
import pytest

from src.experimental import WalkForwardValidator
from src.model import XGBoostStockPredictor
from src.pipeline.stacked_pipeline import StackedModelTrainingPipeline
from tests.helpers import fast_xgb_params, fast_lgb_params, make_ohlcv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dataset(n: int = 400, seed: int = 0) -> tuple[pd.DataFrame, pd.Series, pd.Index, pd.Series]:
    """Return a small deterministic (x, y, dates, prices) tuple for validator tests."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    x = pd.DataFrame(
        {
            "f1": rng.normal(0, 1, n),
            "f2": rng.normal(0, 1, n),
            "f3": rng.uniform(0, 1, n),
        },
        index=dates,
    )
    y = pd.Series(rng.normal(0, 0.02, n), index=dates, name="target")
    prices = pd.Series(100.0 + np.cumsum(rng.normal(0, 1, n)), index=dates, name="price")
    return x, y, dates, prices


def _fast_xgb_factory(params):
    """Factory compatible with WalkForwardValidator.validate() signature."""
    return XGBoostStockPredictor(params=fast_xgb_params())


class _StubDownloader:
    def get_sector(self, symbol): return "Technology"
    def get_industry(self, symbol): return "Software"
    def get_beta(self, symbol): return 1.0
    def get_avg_volume(self, symbol): return 1_000_000.0
    def get_market_cap(self, symbol): return 1_000_000_000.0


def _build_pipeline(monkeypatch, *, use_walk_forward=False, window_type="expanding") -> StackedModelTrainingPipeline:
    symbols = ["AAPL", "MSFT"]
    intervals = ["1d", "1h"]
    pipeline = StackedModelTrainingPipeline(
        intervals=intervals,
        symbols=symbols,
        test_size=0.2,
        use_walk_forward=use_walk_forward,
        walk_forward_window_type=window_type,
        walk_forward_train_window=80,
        walk_forward_test_window=20,
        walk_forward_retrain_freq=20,
    )
    pipeline._xgb_params = {**fast_xgb_params(), "n_estimators": 5}
    pipeline._lgb_params = {**fast_lgb_params(), "n_estimators": 5}

    stock_data = {}
    for i, interval in enumerate(intervals):
        stock_data[interval] = {s: make_ohlcv(n=200, seed=j + i * 100) for j, s in enumerate(symbols)}
    pipeline._stock_data = stock_data
    pipeline._downloader = _StubDownloader()

    monkeypatch.setattr("src.pipeline.stacked_pipeline.save_trained_model", lambda *a, **kw: None)
    return pipeline


# ---------------------------------------------------------------------------
# WalkForwardValidator — unit tests
# ---------------------------------------------------------------------------


class TestWalkForwardValidatorSplit:
    def test_expanding_window_produces_folds(self):
        x, y, dates, _ = _make_dataset()
        v = WalkForwardValidator(train_window=200, test_window=40, retrain_frequency=40, window_type="expanding")
        splits = v.split(x, dates)
        assert len(splits) >= 1
        # Expanding: first fold train starts at 0 always
        for s in splits:
            assert s["train_indices"][0] == 0

    def test_sliding_window_advances_start(self):
        x, y, dates, _ = _make_dataset()
        v = WalkForwardValidator(train_window=100, test_window=20, retrain_frequency=20, window_type="sliding")
        splits = v.split(x, dates)
        assert len(splits) >= 2
        # Sliding: each successive fold's train start must be further along
        starts = [s["train_indices"][0] for s in splits]
        assert starts == sorted(starts)
        assert starts[1] > starts[0]

    def test_fold_indices_are_non_overlapping(self):
        x, y, dates, _ = _make_dataset(n=300)
        v = WalkForwardValidator(train_window=100, test_window=30, retrain_frequency=30, window_type="expanding")
        splits = v.split(x, dates)
        for s in splits:
            train_set = set(s["train_indices"])
            test_set = set(s["test_indices"])
            assert train_set.isdisjoint(test_set), "Train and test indices must not overlap"

    def test_test_indices_follow_train(self):
        x, y, dates, _ = _make_dataset(n=300)
        v = WalkForwardValidator(train_window=100, test_window=30, retrain_frequency=30)
        splits = v.split(x, dates)
        for s in splits:
            assert max(s["train_indices"]) < min(s["test_indices"]), "Test period must follow training period"

    def test_insufficient_data_raises(self):
        x, y, dates, _ = _make_dataset(n=10)
        v = WalkForwardValidator(train_window=100, test_window=20)
        with pytest.raises(ValueError, match="Insufficient data"):
            v.split(x, dates)

    def test_invalid_window_type_raises(self):
        with pytest.raises(ValueError, match="window_type"):
            WalkForwardValidator(window_type="unknown")

    def test_fold_numbering_starts_at_one(self):
        x, y, dates, _ = _make_dataset()
        v = WalkForwardValidator(train_window=100, test_window=20, retrain_frequency=20)
        splits = v.split(x, dates)
        assert splits[0]["fold"] == 1


class TestWalkForwardValidatorMetrics:
    def test_directional_accuracy_perfect(self):
        y = np.array([1.0, -1.0, 0.5])
        pred = np.array([2.0, -0.5, 0.1])
        acc = WalkForwardValidator._directional_accuracy(y, pred)
        assert acc == pytest.approx(100.0)

    def test_directional_accuracy_none(self):
        y = np.array([1.0, 1.0, 1.0])
        pred = np.array([-1.0, -1.0, -1.0])
        acc = WalkForwardValidator._directional_accuracy(y, pred)
        assert acc == pytest.approx(0.0)

    def test_sharpe_zero_when_no_variance(self):
        y = np.zeros(20)
        pred = np.ones(20)
        sharpe = WalkForwardValidator._sharpe_ratio(y, pred)
        assert sharpe == pytest.approx(0.0)

    def test_sharpe_positive_when_strategy_profitable(self):
        rng = np.random.default_rng(1)
        y = rng.normal(0.01, 0.02, 50)   # mostly positive returns
        pred = np.ones(50) * 0.01        # always predict positive → always long
        sharpe = WalkForwardValidator._sharpe_ratio(y, pred)
        assert sharpe > 0.0

    def test_degradation_flag_triggered(self):
        fold_metrics = [
            {"r2": 0.4},
            {"r2": 0.3},
            {"r2": 0.05},  # large drop
        ]
        assert WalkForwardValidator._flag_degradation(fold_metrics, metric="r2") is True

    def test_degradation_flag_not_triggered_on_stable(self):
        fold_metrics = [
            {"r2": 0.4},
            {"r2": 0.38},
            {"r2": 0.39},
        ]
        assert WalkForwardValidator._flag_degradation(fold_metrics, metric="r2") is False

    def test_degradation_flag_with_single_fold(self):
        assert WalkForwardValidator._flag_degradation([{"r2": 0.5}]) is False


class TestWalkForwardValidatorValidate:
    def test_validate_returns_required_keys(self):
        x, y, dates, prices = _make_dataset(n=300)
        v = WalkForwardValidator(train_window=100, test_window=30, retrain_frequency=30)
        results = v.validate(x, y, dates, prices, _fast_xgb_factory)
        required_keys = ("predictions", "actuals", "dates", "prices", "fold_boundaries", "fold_metrics", "metrics",
                         "regime_sensitive")
        for key in required_keys:
            assert key in results, f"Missing key: {key}"

    def test_metrics_contain_all_fields(self):
        x, y, dates, prices = _make_dataset(n=300)
        v = WalkForwardValidator(train_window=100, test_window=30, retrain_frequency=30)
        results = v.validate(x, y, dates, prices, _fast_xgb_factory)
        for field in ("r2", "rmse", "mae", "mse", "directional_accuracy", "sharpe"):
            assert field in results["metrics"], f"Missing metric: {field}"
            assert np.isfinite(results["metrics"][field])

    def test_per_fold_metrics_contain_all_fields(self):
        x, y, dates, prices = _make_dataset(n=300)
        v = WalkForwardValidator(train_window=100, test_window=30, retrain_frequency=30)
        results = v.validate(x, y, dates, prices, _fast_xgb_factory)
        for fold in results["fold_metrics"]:
            for field in ("fold", "r2", "rmse", "mae", "directional_accuracy", "sharpe", "n_train", "n_test"):
                assert field in fold

    def test_predictions_and_actuals_same_length(self):
        x, y, dates, prices = _make_dataset(n=300)
        v = WalkForwardValidator(train_window=100, test_window=30, retrain_frequency=30)
        results = v.validate(x, y, dates, prices, _fast_xgb_factory)
        assert len(results["predictions"]) == len(results["actuals"])

    def test_regime_sensitive_is_bool(self):
        x, y, dates, prices = _make_dataset(n=300)
        v = WalkForwardValidator(train_window=100, test_window=30, retrain_frequency=30)
        results = v.validate(x, y, dates, prices, _fast_xgb_factory)
        assert isinstance(results["regime_sensitive"], bool)

    def test_fold_results_stored(self):
        x, y, dates, prices = _make_dataset(n=300)
        v = WalkForwardValidator(train_window=100, test_window=30, retrain_frequency=30)
        results = v.validate(x, y, dates, prices, _fast_xgb_factory)
        assert len(v.fold_results) == len(results["fold_metrics"])

    def test_sliding_window_validate(self):
        x, y, dates, prices = _make_dataset(n=300)
        v = WalkForwardValidator(train_window=100, test_window=30, retrain_frequency=30, window_type="sliding")
        results = v.validate(x, y, dates, prices, _fast_xgb_factory)
        assert len(results["fold_metrics"]) >= 1


# ---------------------------------------------------------------------------
# Pipeline integration tests
# ---------------------------------------------------------------------------


class TestPipelineWalkForwardIntegration:
    def test_pipeline_walk_forward_disabled_by_default(self, monkeypatch):
        pipeline = _build_pipeline(monkeypatch)
        assert pipeline.use_walk_forward is False
        assert pipeline._walk_forward_results is None

    def test_pipeline_stores_walk_forward_results_after_train(self, monkeypatch):
        pipeline = _build_pipeline(monkeypatch, use_walk_forward=True)
        pipeline.prepare_features()
        pipeline.train()
        assert pipeline._walk_forward_results is not None
        assert "fold_metrics" in pipeline._walk_forward_results

    def test_walk_forward_validate_raises_before_prepare(self, monkeypatch):
        pipeline = _build_pipeline(monkeypatch, use_walk_forward=True)
        with pytest.raises(Exception, match="prepare_features"):
            pipeline.walk_forward_validate()

    def test_walk_forward_validate_standalone(self, monkeypatch):
        pipeline = _build_pipeline(monkeypatch)
        pipeline.prepare_features()
        results = pipeline.walk_forward_validate()
        assert "fold_metrics" in results
        assert "metrics" in results
        assert len(results["fold_metrics"]) >= 1

    def test_walk_forward_sliding_window_in_pipeline(self, monkeypatch):
        pipeline = _build_pipeline(monkeypatch, use_walk_forward=True, window_type="sliding")
        pipeline.prepare_features()
        pipeline.train()
        assert pipeline._walk_forward_results is not None

    def test_pipeline_train_succeeds_without_walk_forward(self, monkeypatch):
        pipeline = _build_pipeline(monkeypatch, use_walk_forward=False)
        pipeline.prepare_features()
        pipeline.train()
        assert pipeline._walk_forward_results is None

    def test_walk_forward_results_have_valid_metrics(self, monkeypatch):
        pipeline = _build_pipeline(monkeypatch, use_walk_forward=True)
        pipeline.prepare_features()
        pipeline.train()
        metrics = pipeline._walk_forward_results["metrics"]
        for field in ("r2", "rmse", "mae", "directional_accuracy", "sharpe"):
            assert np.isfinite(metrics[field]), f"Metric '{field}' is not finite: {metrics[field]}"
