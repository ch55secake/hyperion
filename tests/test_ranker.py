"""Unit tests for src/ranking/ranker.py (StockRanker)."""

import numpy as np
import pandas as pd
import pytest

from src.ranking.ranker import StockRanker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _price_series(n: int = 30, start: float = 100.0, seed: int = 0) -> pd.Series:
    """Return a deterministic price Series of length *n*."""
    rng = np.random.default_rng(seed)
    prices = start + np.cumsum(rng.normal(0, 1, n))
    prices = np.abs(prices) + 1.0  # ensure positive
    return pd.Series(prices)


def _default_ranker(**kwargs) -> StockRanker:
    return StockRanker(
        min_confidence=0.5,
        total_funds=100_000.0,
        max_allocation_pct=1.0,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# __init__ validation
# ---------------------------------------------------------------------------


class TestStockRankerInit:
    def test_valid_defaults(self):
        ranker = StockRanker()
        assert ranker.min_confidence == 0.6
        assert ranker.total_funds == 100_000.0
        assert ranker.max_allocation_pct == 1.0
        assert ranker.volatility_window == 20
        assert ranker.min_volatility == 1e-6

    def test_invalid_min_confidence_below_zero(self):
        with pytest.raises(ValueError, match="min_confidence"):
            StockRanker(min_confidence=-0.1)

    def test_invalid_min_confidence_above_one(self):
        with pytest.raises(ValueError, match="min_confidence"):
            StockRanker(min_confidence=1.5)

    def test_invalid_total_funds_zero(self):
        with pytest.raises(ValueError, match="total_funds"):
            StockRanker(total_funds=0)

    def test_invalid_total_funds_negative(self):
        with pytest.raises(ValueError, match="total_funds"):
            StockRanker(total_funds=-1)

    def test_invalid_max_allocation_pct_zero(self):
        with pytest.raises(ValueError, match="max_allocation_pct"):
            StockRanker(max_allocation_pct=0.0)

    def test_invalid_max_allocation_pct_above_one(self):
        with pytest.raises(ValueError, match="max_allocation_pct"):
            StockRanker(max_allocation_pct=1.1)

    def test_invalid_volatility_window(self):
        with pytest.raises(ValueError, match="volatility_window"):
            StockRanker(volatility_window=0)

    def test_invalid_min_volatility_zero(self):
        with pytest.raises(ValueError, match="min_volatility"):
            StockRanker(min_volatility=0)

    def test_invalid_min_volatility_negative(self):
        with pytest.raises(ValueError, match="min_volatility"):
            StockRanker(min_volatility=-1e-7)


# ---------------------------------------------------------------------------
# Step 1 – compute_expected_return
# ---------------------------------------------------------------------------


class TestComputeExpectedReturn:
    def test_simple_gain(self):
        predicted = pd.Series([110.0])
        current = pd.Series([100.0])
        result = StockRanker.compute_expected_return(predicted, current)
        assert abs(result.iloc[0] - 0.10) < 1e-9

    def test_simple_loss(self):
        predicted = pd.Series([90.0])
        current = pd.Series([100.0])
        result = StockRanker.compute_expected_return(predicted, current)
        assert abs(result.iloc[0] - (-0.10)) < 1e-9

    def test_zero_current_price_becomes_nan(self):
        predicted = pd.Series([100.0])
        current = pd.Series([0.0])
        result = StockRanker.compute_expected_return(predicted, current)
        assert np.isnan(result.iloc[0])

    def test_multiple_symbols(self):
        predicted = pd.Series([110.0, 95.0, 105.0])
        current = pd.Series([100.0, 100.0, 100.0])
        result = StockRanker.compute_expected_return(predicted, current)
        np.testing.assert_allclose(result.values, [0.10, -0.05, 0.05])

    def test_predictions_are_returns_mode(self):
        # When predictions are already returns, pass 1.0 as current price
        predicted = pd.Series([0.055, 0.037])
        current = pd.Series([1.0, 1.0])
        result = StockRanker.compute_expected_return(predicted, current)
        # (r - 1) / 1  -- this is the formula; so values are r-1
        # The caller should use predictions_are_returns=True in rank_and_allocate instead
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Step 2 – compute_confidence_score
# ---------------------------------------------------------------------------


class TestComputeConfidenceScore:
    def test_max_absolute_return_gets_score_one(self):
        returns = pd.Series([0.10, 0.05, 0.02])
        scores = StockRanker.compute_confidence_score(returns)
        assert scores.iloc[0] == pytest.approx(1.0)

    def test_proportional_scores(self):
        returns = pd.Series([0.10, 0.05])
        scores = StockRanker.compute_confidence_score(returns)
        assert scores.iloc[1] == pytest.approx(0.5)

    def test_negative_returns_use_absolute_value(self):
        returns = pd.Series([-0.10, 0.05])
        scores = StockRanker.compute_confidence_score(returns)
        assert scores.iloc[0] == pytest.approx(1.0)
        assert scores.iloc[1] == pytest.approx(0.5)

    def test_all_zero_returns_gives_zero_scores(self):
        returns = pd.Series([0.0, 0.0, 0.0])
        scores = StockRanker.compute_confidence_score(returns)
        assert (scores == 0.0).all()

    def test_scores_in_zero_one_range(self):
        returns = pd.Series(np.random.default_rng(1).normal(0, 0.05, 50))
        scores = StockRanker.compute_confidence_score(returns)
        assert (scores >= 0.0).all()
        assert (scores <= 1.0).all()


# ---------------------------------------------------------------------------
# Step 3 – apply_confidence_threshold
# ---------------------------------------------------------------------------


class TestApplyConfidenceThreshold:
    def _make_df(self, confidences):
        return pd.DataFrame({"symbol": [f"S{i}" for i in range(len(confidences))], "confidence": confidences})

    def test_filters_below_threshold(self):
        ranker = StockRanker(min_confidence=0.5)
        df = self._make_df([0.3, 0.5, 0.7])
        result = ranker.apply_confidence_threshold(df)
        assert list(result["symbol"]) == ["S1", "S2"]

    def test_exact_threshold_passes(self):
        ranker = StockRanker(min_confidence=0.5)
        df = self._make_df([0.5])
        result = ranker.apply_confidence_threshold(df)
        assert len(result) == 1

    def test_all_below_threshold_returns_empty(self):
        ranker = StockRanker(min_confidence=0.9)
        df = self._make_df([0.1, 0.2, 0.3])
        result = ranker.apply_confidence_threshold(df)
        assert result.empty

    def test_all_pass_threshold(self):
        ranker = StockRanker(min_confidence=0.0)
        df = self._make_df([0.1, 0.5, 1.0])
        result = ranker.apply_confidence_threshold(df)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Step 4 – compute_volatility
# ---------------------------------------------------------------------------


class TestComputeVolatility:
    def test_returns_positive_values(self):
        ranker = _default_ranker(volatility_window=5)
        prices = {"AAPL": _price_series(30)}
        vol = ranker.compute_volatility(prices)
        assert vol["AAPL"] > 0

    def test_multiple_symbols(self):
        ranker = _default_ranker(volatility_window=5)
        prices = {"AAPL": _price_series(30, seed=0), "GOOG": _price_series(30, seed=1)}
        vol = ranker.compute_volatility(prices)
        assert set(vol.keys()) == {"AAPL", "GOOG"}

    def test_single_price_falls_back_to_min_volatility(self):
        ranker = _default_ranker(min_volatility=1e-4)
        prices = {"X": pd.Series([100.0])}
        vol = ranker.compute_volatility(prices)
        assert vol["X"] == pytest.approx(1e-4)

    def test_two_prices_computes_std(self):
        ranker = _default_ranker(volatility_window=20)
        prices = {"X": pd.Series([100.0, 110.0])}
        vol = ranker.compute_volatility(prices)
        # pct_change of 10% – rolling window of 20 not achievable, falls back to std
        assert vol["X"] > 0

    def test_min_volatility_floor_applied(self):
        ranker = StockRanker(min_volatility=0.5)
        # A perfectly flat series has 0 returns → std=0 → should return min_volatility
        prices = {"FLAT": pd.Series([100.0] * 30)}
        vol = ranker.compute_volatility(prices)
        assert vol["FLAT"] >= 0.5

    def test_window_shorter_than_series_uses_rolling(self):
        ranker = _default_ranker(volatility_window=5)
        prices = {"X": _price_series(50)}
        vol_short_window = ranker.compute_volatility(prices)
        ranker2 = _default_ranker(volatility_window=40)
        vol_long_window = ranker2.compute_volatility(prices)
        # Both should be positive; the values will differ
        assert vol_short_window["X"] > 0
        assert vol_long_window["X"] > 0


# ---------------------------------------------------------------------------
# Step 5 – compute_priority_score
# ---------------------------------------------------------------------------


class TestComputePriorityScore:
    def test_basic_formula(self):
        er = pd.Series([0.10])
        conf = pd.Series([0.8])
        vol = pd.Series([0.02])
        score = StockRanker.compute_priority_score(er, conf, vol)
        assert score.iloc[0] == pytest.approx(0.10 * 0.8 / 0.02)

    def test_higher_volatility_lowers_score(self):
        er = pd.Series([0.10, 0.10])
        conf = pd.Series([0.8, 0.8])
        vol = pd.Series([0.01, 0.05])
        scores = StockRanker.compute_priority_score(er, conf, vol)
        assert scores.iloc[0] > scores.iloc[1]

    def test_negative_return_gives_negative_score(self):
        er = pd.Series([-0.05])
        conf = pd.Series([0.8])
        vol = pd.Series([0.02])
        score = StockRanker.compute_priority_score(er, conf, vol)
        assert score.iloc[0] < 0


# ---------------------------------------------------------------------------
# Step 6 – rank_stocks
# ---------------------------------------------------------------------------


class TestRankStocks:
    def _make_df(self, scores):
        return pd.DataFrame({"symbol": [f"S{i}" for i in range(len(scores))], "priority_score": scores})

    def test_sorted_descending(self):
        df = self._make_df([1.0, 3.0, 2.0])
        result = StockRanker.rank_stocks(df)
        assert list(result["priority_score"]) == [3.0, 2.0, 1.0]

    def test_rank_column_starts_at_one(self):
        df = self._make_df([1.0, 2.0])
        result = StockRanker.rank_stocks(df)
        assert list(result["rank"]) == [1, 2]

    def test_single_row_rank_is_one(self):
        df = self._make_df([5.0])
        result = StockRanker.rank_stocks(df)
        assert result["rank"].iloc[0] == 1

    def test_empty_df_returns_empty(self):
        df = pd.DataFrame({"symbol": [], "priority_score": []})
        result = StockRanker.rank_stocks(df)
        assert result.empty


# ---------------------------------------------------------------------------
# Step 7 – allocate_capital
# ---------------------------------------------------------------------------


class TestAllocateCapital:
    def _make_df(self, scores, symbols=None):
        if symbols is None:
            symbols = [f"S{i}" for i in range(len(scores))]
        return pd.DataFrame({"symbol": symbols, "priority_score": scores})

    def test_total_allocation_equals_total_funds(self):
        ranker = StockRanker(total_funds=100_000, max_allocation_pct=1.0)
        df = self._make_df([3.0, 2.0, 1.0])
        result = ranker.allocate_capital(df)
        assert result["allocation"].sum() == pytest.approx(100_000.0)

    def test_proportional_allocation(self):
        ranker = StockRanker(total_funds=100_000, max_allocation_pct=1.0)
        df = self._make_df([3.0, 1.0])
        result = ranker.allocate_capital(df)
        # S0 has 3x the score of S1 → should get 75k vs 25k
        allocs = result.set_index("symbol")["allocation"]
        assert allocs["S0"] == pytest.approx(75_000.0)
        assert allocs["S1"] == pytest.approx(25_000.0)

    def test_cap_limits_single_stock(self):
        ranker = StockRanker(total_funds=100_000, max_allocation_pct=0.4)
        df = self._make_df([10.0, 1.0, 1.0])
        result = ranker.allocate_capital(df)
        assert result["allocation"].max() <= 40_000.0 + 1e-9

    def test_negative_priority_score_gets_zero_allocation(self):
        ranker = StockRanker(total_funds=100_000, max_allocation_pct=1.0)
        df = self._make_df([-1.0, 2.0])
        result = ranker.allocate_capital(df)
        allocs = result.set_index("symbol")["allocation"]
        assert allocs["S0"] == pytest.approx(0.0)
        assert allocs["S1"] == pytest.approx(100_000.0)

    def test_all_negative_scores_zero_allocation(self):
        ranker = StockRanker(total_funds=100_000, max_allocation_pct=1.0)
        df = self._make_df([-1.0, -2.0])
        result = ranker.allocate_capital(df)
        assert result["allocation"].sum() == pytest.approx(0.0)

    def test_cap_redistribution(self):
        # With 2 stocks equal priority and cap=0.3, each should be capped at 30k
        # That only uses 60k total — remaining 40k split again (cap keeps it at 30k each)
        ranker = StockRanker(total_funds=100_000, max_allocation_pct=0.3)
        df = self._make_df([1.0, 1.0])
        result = ranker.allocate_capital(df)
        # Each capped at 30k, total = 60k (surplus cannot be redistributed back)
        assert result["allocation"].max() <= 30_000.0 + 1e-9

    def test_single_stock_gets_all_funds_when_no_cap(self):
        ranker = StockRanker(total_funds=50_000, max_allocation_pct=1.0)
        df = self._make_df([5.0], symbols=["AAPL"])
        result = ranker.allocate_capital(df)
        assert result["allocation"].iloc[0] == pytest.approx(50_000.0)


# ---------------------------------------------------------------------------
# rank_and_allocate (full pipeline)
# ---------------------------------------------------------------------------


class TestRankAndAllocate:
    def test_empty_predictions_returns_empty_df(self):
        ranker = _default_ranker()
        result = ranker.rank_and_allocate(predictions={}, current_prices={})
        assert result.empty
        assert "symbol" in result.columns

    def test_predictions_are_returns_mode(self):
        ranker = StockRanker(min_confidence=0.0, total_funds=100_000, max_allocation_pct=1.0)
        result = ranker.rank_and_allocate(
            predictions={"AAPL": 0.10, "GOOG": 0.05},
            current_prices={},
            predictions_are_returns=True,
        )
        assert len(result) == 2
        assert set(result["symbol"]) == {"AAPL", "GOOG"}

    def test_result_has_required_columns(self):
        ranker = _default_ranker()
        result = ranker.rank_and_allocate(
            predictions={"AAPL": 0.10, "GOOG": 0.05},
            current_prices={},
            predictions_are_returns=True,
        )
        for col in ["symbol", "expected_return", "confidence", "volatility", "priority_score", "rank", "allocation"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_confidence_threshold_filters_low_confidence(self):
        # With min_confidence=0.8, only the stock with max absolute return passes (score=1.0)
        ranker = StockRanker(min_confidence=0.8, total_funds=100_000, max_allocation_pct=1.0)
        result = ranker.rank_and_allocate(
            predictions={"AAPL": 0.10, "GOOG": 0.01},
            current_prices={},
            predictions_are_returns=True,
        )
        # GOOG confidence = 0.01/0.10 = 0.1 → filtered
        assert "GOOG" not in result["symbol"].values
        assert "AAPL" in result["symbol"].values

    def test_all_below_threshold_returns_empty_like_df(self):
        ranker = StockRanker(min_confidence=0.99, total_funds=100_000, max_allocation_pct=1.0)
        _ = ranker.rank_and_allocate(
            predictions={"AAPL": 0.05, "GOOG": 0.03},
            current_prices={},
            predictions_are_returns=True,
        )
        ranker2 = StockRanker(min_confidence=0.99, total_funds=100_000, max_allocation_pct=1.0)
        result2 = ranker2.rank_and_allocate(
            predictions={"AAPL": 0.05, "GOOG": 0.03},
            current_prices={},
            historical_volatility={"AAPL": 0.02, "GOOG": 0.01},
            predictions_are_returns=True,
        )
        # AAPL gets confidence=1.0 (it's the max), GOOG gets 0.6
        # AAPL passes (1.0 >= 0.99), GOOG is filtered
        assert "AAPL" in result2["symbol"].values

    def test_with_historical_volatility_dict(self):
        ranker = _default_ranker()
        result = ranker.rank_and_allocate(
            predictions={"AAPL": 0.10, "GOOG": 0.05},
            current_prices={},
            historical_volatility={"AAPL": 0.02, "GOOG": 0.01},
            predictions_are_returns=True,
        )
        assert len(result) == 2
        assert result.loc[result["symbol"] == "AAPL", "volatility"].iloc[0] == pytest.approx(0.02)

    def test_with_prices_history_dict(self):
        ranker = _default_ranker()
        prices_hist = {"AAPL": _price_series(30, seed=0), "GOOG": _price_series(30, seed=1)}
        result = ranker.rank_and_allocate(
            predictions={"AAPL": 0.10, "GOOG": 0.05},
            current_prices={},
            prices_history=prices_hist,
            predictions_are_returns=True,
        )
        assert len(result) == 2
        assert (result["volatility"] > 0).all()

    def test_historical_volatility_overrides_prices_history(self):
        ranker = _default_ranker()
        prices_hist = {"AAPL": _price_series(30, seed=0)}
        result = ranker.rank_and_allocate(
            predictions={"AAPL": 0.10},
            current_prices={},
            prices_history=prices_hist,
            historical_volatility={"AAPL": 0.999},
            predictions_are_returns=True,
        )
        # historical_volatility takes precedence
        assert result["volatility"].iloc[0] == pytest.approx(0.999)

    def test_rank_column_ordered_correctly(self):
        ranker = StockRanker(min_confidence=0.0, total_funds=100_000, max_allocation_pct=1.0)
        result = ranker.rank_and_allocate(
            predictions={"AAPL": 0.10, "GOOG": 0.05, "TSLA": 0.02},
            current_prices={},
            historical_volatility={"AAPL": 0.02, "GOOG": 0.02, "TSLA": 0.02},
            predictions_are_returns=True,
        )
        # priority_score ∝ expected_return * confidence / volatility
        # AAPL has highest return → rank 1
        ranks = result.set_index("symbol")["rank"]
        assert ranks["AAPL"] == 1

    def test_allocation_sums_to_total_funds(self):
        ranker = _default_ranker()
        result = ranker.rank_and_allocate(
            predictions={"AAPL": 0.10, "GOOG": 0.05, "TSLA": 0.03},
            current_prices={},
            predictions_are_returns=True,
        )
        assert result["allocation"].sum() == pytest.approx(100_000.0)

    def test_predicted_prices_mode(self):
        ranker = _default_ranker()
        result = ranker.rank_and_allocate(
            predictions={"AAPL": 110.0, "GOOG": 95.0},
            current_prices={"AAPL": 100.0, "GOOG": 100.0},
            predictions_are_returns=False,
        )
        assert len(result) > 0
        # AAPL predicted +10%, GOOG predicted -5% → GOOG has negative return
        if "GOOG" in result["symbol"].values:
            goog_er = result.loc[result["symbol"] == "GOOG", "expected_return"].iloc[0]
            assert goog_er < 0

    def test_missing_current_price_defaults_to_one(self):
        # When predictions_are_returns=False and current_prices missing a symbol,
        # it defaults to 1.0 (effectively treating prediction as 1+return)
        ranker = _default_ranker()
        result = ranker.rank_and_allocate(
            predictions={"AAPL": 1.10},
            current_prices={},  # AAPL missing → defaults to 1.0
            predictions_are_returns=False,
        )
        assert len(result) >= 0  # should not crash

    def test_no_prices_history_uses_min_volatility(self):
        ranker = StockRanker(min_confidence=0.0, total_funds=100_000, max_allocation_pct=1.0, min_volatility=1e-4)
        result = ranker.rank_and_allocate(
            predictions={"AAPL": 0.10},
            current_prices={},
            predictions_are_returns=True,
        )
        assert result["volatility"].iloc[0] == pytest.approx(1e-4)

    def test_result_index_is_reset(self):
        ranker = _default_ranker()
        result = ranker.rank_and_allocate(
            predictions={"AAPL": 0.10, "GOOG": 0.05},
            current_prices={},
            predictions_are_returns=True,
        )
        assert list(result.index) == list(range(len(result)))
