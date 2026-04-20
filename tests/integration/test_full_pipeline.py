"""
True end-to-end integration tests for all training pipelines.

Covers the complete sequence for each pipeline class:
  read_tickers -> download_data -> prepare_features -> train -> simulate -> rank_and_allocate

StockDataDownloader is mocked so no network calls are made.  All other pipeline
stages execute real production code against deterministic synthetic OHLCV fixtures.

Pipelines under test:
  - StackedModelTrainingPipeline      (XGBoost + LightGBM stacked, multi-interval)
  - SingleModelTrainingPipeline       (single XGBoost model, single interval)
  - TimeSeriesStackedModelTrainingPipeline  (XGBoost + LightGBM + CatBoost meta-stacker)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.pipeline.single_pipeline import SingleModelTrainingPipeline
from src.pipeline.stacked_pipeline import StackedModelTrainingPipeline
from src.pipeline.time_series_stacked_pipeline import TimeSeriesStackedModelTrainingPipeline
from tests.helpers import fast_cat_params, fast_lgb_params, fast_xgb_params, make_ohlcv

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
_INTERVALS = ["1d", "1h"]
_N_ROWS = 120  # enough rows for train/val/test splits


# ---------------------------------------------------------------------------
# Shared mock helpers
# ---------------------------------------------------------------------------


def _make_mock_downloader(symbols: list[str], n: int, seed_offset: int = 0) -> MagicMock:
    """Return a MagicMock that behaves like StockDataDownloader for one interval."""
    stock_data = {symbol: make_ohlcv(n=n, seed=j + seed_offset) for j, symbol in enumerate(symbols)}
    mock = MagicMock()
    mock.download_data.return_value = (stock_data, [])
    mock.get_sector.return_value = "Technology"
    mock.get_industry.return_value = "Software"
    mock.get_beta.return_value = 1.0
    mock.get_avg_volume.return_value = 1_000_000.0
    mock.get_market_cap.return_value = 1_000_000_000.0
    return mock


# ---------------------------------------------------------------------------
# StackedModelTrainingPipeline fixture (multi-interval download)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def full_pipeline():
    """
    Build and run the complete StackedModelTrainingPipeline sequence using synthetic data.

    simulate() is called without a strategy_name so that every registered strategy
    is exercised in a single pass.
    """
    pipeline = StackedModelTrainingPipeline(
        intervals=_INTERVALS,
        symbols=_TICKERS,
        test_size=0.2,
        val_size=0.1,
    )
    pipeline._xgb_params = {**fast_xgb_params(), "n_estimators": 10}
    pipeline._lgb_params = {**fast_lgb_params(), "n_estimators": 10}

    # One mock downloader per interval; download_data() instantiates them in order.
    interval_mocks = {
        interval: _make_mock_downloader(_TICKERS, _N_ROWS, i * 100) for i, interval in enumerate(_INTERVALS)
    }
    call_count: list[int] = [0]

    def _downloader_factory(*args, **kwargs):
        interval = _INTERVALS[call_count[0] % len(_INTERVALS)]
        call_count[0] += 1
        return interval_mocks[interval]

    with (
        patch("src.pipeline.stacked_pipeline.StockDataDownloader", side_effect=_downloader_factory),
        patch("src.pipeline.stacked_pipeline.save_trained_model", return_value=None),
    ):
        pipeline.read_tickers().download_data().prepare_features().train()
        # Run all registered strategies (no strategy_name = all strategies)
        pipeline.simulate()
        return pipeline


# ---------------------------------------------------------------------------
# SingleModelTrainingPipeline fixture (single-interval download)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def full_single_pipeline():
    """
    Build and run the complete SingleModelTrainingPipeline (XGBoost) sequence.

    Uses the base-class download_data() which creates one StockDataDownloader for
    a single interval.
    """
    pipeline = SingleModelTrainingPipeline(
        model_type="xgboost",
        symbols=_TICKERS,
        test_size=0.2,
    )
    pipeline._model_params = {**fast_xgb_params(), "n_estimators": 10}

    mock_downloader = _make_mock_downloader(_TICKERS, _N_ROWS, seed_offset=200)

    with (
        patch("src.pipeline.base_pipeline.StockDataDownloader", return_value=mock_downloader),
        patch("src.pipeline.single_pipeline.save_trained_model", return_value=None),
    ):
        pipeline.read_tickers().download_data().prepare_features().train()
        # Run all registered strategies (no strategy_name = all strategies)
        pipeline.simulate()
        return pipeline


# ---------------------------------------------------------------------------
# TimeSeriesStackedModelTrainingPipeline fixture (single-interval download)
# ---------------------------------------------------------------------------

# TSSP uses TimeSeriesSplit internally, which requires unique timestamps per row.
# The base prepare_features() produces duplicate timestamps when multiple tickers
# are concatenated.  A single ticker avoids that limitation.
_TS_TICKERS = ["AAPL"]
_TS_N_ROWS = 200  # more rows to ensure enough samples for n_splits=5 CV folds


@pytest.fixture(scope="module")
def full_ts_stacked_pipeline():
    """
    Build and run the complete TimeSeriesStackedModelTrainingPipeline sequence.

    Uses the base-class download_data() which creates one StockDataDownloader.
    XGBoost, LightGBM, and CatBoost params are set to minimal values to keep
    the test fast while still exercising the full stacking logic.

    A single ticker is used because TimeSeriesStacker performs TimeSeriesSplit
    over the index; concatenating multiple tickers with overlapping timestamps
    produces duplicate index entries that confuse the splitter.
    """
    pipeline = TimeSeriesStackedModelTrainingPipeline(
        symbols=_TS_TICKERS,
        test_size=0.2,
    )
    pipeline._xgb_params = {**fast_xgb_params(), "n_estimators": 5}
    pipeline._lgb_params = {**fast_lgb_params(), "n_estimators": 5}
    pipeline._cat_params = fast_cat_params()

    mock_downloader = _make_mock_downloader(_TS_TICKERS, _TS_N_ROWS, seed_offset=300)

    with (
        patch("src.pipeline.base_pipeline.StockDataDownloader", return_value=mock_downloader),
        patch("src.pipeline.time_series_stacked_pipeline.save_trained_model", return_value=None),
    ):
        pipeline.read_tickers().download_data().prepare_features().train()
        # Run all registered strategies (no strategy_name = all strategies)
        pipeline.simulate()
        return pipeline


# ---------------------------------------------------------------------------
# StackedModelTrainingPipeline tests
# ---------------------------------------------------------------------------


class TestFullPipelineAlignment:
    """Validate that StackedModelTrainingPipeline.train() leaves all test arrays properly aligned."""

    def test_predictions_exist(self, full_pipeline):
        predictions = full_pipeline._test_results.get("predictions")
        assert predictions is not None, "_test_results must contain 'predictions' after train()"
        assert len(predictions) > 0, "predictions array must be non-empty"

    def test_predictions_symbols_dates_prices_same_length(self, full_pipeline):
        predictions = full_pipeline._test_results["predictions"]
        n_preds = len(predictions)
        n_symbols = len(full_pipeline._symbols_test)
        n_dates = len(full_pipeline._dates_test)
        n_prices = len(full_pipeline._prices_test)

        assert n_preds == n_symbols, f"predictions length {n_preds} != symbols length {n_symbols}"
        assert n_preds == n_dates, f"predictions length {n_preds} != dates length {n_dates}"
        assert n_preds == n_prices, f"predictions length {n_preds} != prices length {n_prices}"

    def test_all_tickers_represented_in_test_set(self, full_pipeline):
        unique_syms = set(full_pipeline._symbols_test.unique())
        for ticker in _TICKERS:
            assert ticker in unique_syms, f"{ticker} missing from test set symbols"

    def test_predictions_are_finite(self, full_pipeline):
        predictions = full_pipeline._test_results["predictions"]
        assert np.all(np.isfinite(predictions)), "predictions contain non-finite values"

    def test_test_metrics_present_and_finite(self, full_pipeline):
        for key in ("r2", "rmse", "mae"):
            val = full_pipeline._test_results[key]
            assert np.isfinite(val), f"metric '{key}' is not finite: {val}"


class TestFullPipelineSimulate:
    """Validate that StackedModelTrainingPipeline.simulate() completes for all strategies."""

    def test_simulate_does_not_raise(self, full_pipeline):
        assert full_pipeline is not None

    def test_x_test_dict_populated(self, full_pipeline):
        assert full_pipeline._x_test_dict is not None, "_x_test_dict should be set after train()"
        for interval in _INTERVALS:
            assert interval in full_pipeline._x_test_dict, f"interval '{interval}' missing from _x_test_dict"
            assert len(full_pipeline._x_test_dict[interval]) > 0


class TestFullPipelineRankAndAllocate:
    """Validate that rank_and_allocate() returns a well-formed DataFrame."""

    @pytest.fixture(scope="class")
    def ranked(self, full_pipeline):
        return full_pipeline.rank_and_allocate(min_confidence=0.0)

    def test_returns_dataframe(self, ranked):
        assert isinstance(ranked, pd.DataFrame), "rank_and_allocate() must return a DataFrame"

    def test_dataframe_is_non_empty(self, ranked):
        assert not ranked.empty, "rank_and_allocate() returned an empty DataFrame"

    def test_expected_columns_present(self, ranked):
        required = {"symbol", "expected_return", "confidence", "volatility", "priority_score", "rank", "allocation"}
        missing = required - set(ranked.columns)
        assert not missing, f"rank_and_allocate() DataFrame missing columns: {missing}"

    def test_allocation_sums_to_positive(self, ranked):
        assert ranked["allocation"].sum() > 0, "total allocated capital must be positive"

    def test_ranks_are_unique_integers(self, ranked):
        ranks = ranked["rank"].tolist()
        assert len(ranks) == len(set(ranks)), "rank values must be unique"
        assert all(isinstance(r, (int, np.integer)) for r in ranks), "rank values must be integers"


# ---------------------------------------------------------------------------
# SingleModelTrainingPipeline tests
# ---------------------------------------------------------------------------


class TestSinglePipelineAlignment:
    """Validate that SingleModelTrainingPipeline.train() leaves all test arrays properly aligned."""

    def test_predictions_exist(self, full_single_pipeline):
        predictions = full_single_pipeline._test_results.get("predictions")
        assert predictions is not None, "_test_results must contain 'predictions' after train()"
        assert len(predictions) > 0, "predictions array must be non-empty"

    def test_predictions_symbols_dates_prices_same_length(self, full_single_pipeline):
        predictions = full_single_pipeline._test_results["predictions"]
        n_preds = len(predictions)
        n_symbols = len(full_single_pipeline._symbols_test)
        n_dates = len(full_single_pipeline._dates_test)
        n_prices = len(full_single_pipeline._prices_test)

        assert n_preds == n_symbols, f"predictions length {n_preds} != symbols length {n_symbols}"
        assert n_preds == n_dates, f"predictions length {n_preds} != dates length {n_dates}"
        assert n_preds == n_prices, f"predictions length {n_preds} != prices length {n_prices}"

    def test_all_tickers_represented_in_test_set(self, full_single_pipeline):
        unique_syms = set(full_single_pipeline._symbols_test.unique())
        for ticker in _TICKERS:
            assert ticker in unique_syms, f"{ticker} missing from test set symbols"

    def test_predictions_are_finite(self, full_single_pipeline):
        predictions = full_single_pipeline._test_results["predictions"]
        assert np.all(np.isfinite(predictions)), "predictions contain non-finite values"

    def test_test_metrics_present_and_finite(self, full_single_pipeline):
        for key in ("r2", "rmse", "mae"):
            val = full_single_pipeline._test_results[key]
            assert np.isfinite(val), f"metric '{key}' is not finite: {val}"

    def test_x_test_dict_has_daily_key(self, full_single_pipeline):
        assert full_single_pipeline._x_test_dict is not None
        assert "daily" in full_single_pipeline._x_test_dict
        assert len(full_single_pipeline._x_test_dict["daily"]) > 0


class TestSinglePipelineSimulate:
    """Validate that SingleModelTrainingPipeline.simulate() completes for all strategies."""

    def test_simulate_does_not_raise(self, full_single_pipeline):
        assert full_single_pipeline is not None


# ---------------------------------------------------------------------------
# TimeSeriesStackedModelTrainingPipeline tests
# ---------------------------------------------------------------------------


class TestTSStackedPipelineAlignment:
    """Validate that TimeSeriesStackedModelTrainingPipeline.train() aligns test arrays."""

    def test_predictions_exist(self, full_ts_stacked_pipeline):
        predictions = full_ts_stacked_pipeline._test_results.get("predictions")
        assert predictions is not None, "_test_results must contain 'predictions' after train()"
        assert len(predictions) > 0, "predictions array must be non-empty"

    def test_predictions_are_finite(self, full_ts_stacked_pipeline):
        predictions = full_ts_stacked_pipeline._test_results["predictions"]
        assert np.all(np.isfinite(predictions)), "predictions contain non-finite values"

    def test_symbols_dates_prices_populated(self, full_ts_stacked_pipeline):
        assert full_ts_stacked_pipeline._symbols_test is not None
        assert full_ts_stacked_pipeline._dates_test is not None
        assert full_ts_stacked_pipeline._prices_test is not None
        assert len(full_ts_stacked_pipeline._symbols_test) > 0

    def test_ticker_represented_in_test_set(self, full_ts_stacked_pipeline):
        unique_syms = set(full_ts_stacked_pipeline._symbols_test.unique())
        for ticker in _TS_TICKERS:
            assert ticker in unique_syms, f"{ticker} missing from test set symbols"


class TestTSStackedPipelineSimulate:
    """Validate that TimeSeriesStackedModelTrainingPipeline.simulate() completes for all strategies."""

    def test_simulate_does_not_raise(self, full_ts_stacked_pipeline):
        assert full_ts_stacked_pipeline is not None
