"""Unit tests for src/feature/regime.py"""

import numpy as np
import pandas as pd

from src.feature.regime import (
    REGIME_BEAR,
    REGIME_BULL,
    REGIME_SIDEWAYS,
    REGIME_VOL_HIGH,
    REGIME_VOL_LOW,
    REGIME_VOL_MEDIUM,
    COL_REGIME_GMM,
    COL_REGIME_TREND,
    COL_REGIME_VOLATILITY,
    classify_regime,
    detect_gmm_regime,
    detect_trend_regime,
    detect_volatility_regime,
)
from tests.helpers import make_ohlcv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_returns(n: int = 200, seed: int = 42) -> pd.Series:
    ohlcv = make_ohlcv(n=n, seed=seed)
    return ohlcv["Close"].pct_change(1).fillna(0)


def _make_close(n: int = 200, seed: int = 42) -> pd.Series:
    return make_ohlcv(n=n, seed=seed)["Close"]


# ---------------------------------------------------------------------------
# detect_volatility_regime
# ---------------------------------------------------------------------------


class TestDetectVolatilityRegime:
    def test_returns_series_same_length(self):
        returns = _make_returns(100)
        result = detect_volatility_regime(returns)
        assert len(result) == len(returns)
        assert result.index.equals(returns.index)

    def test_only_valid_labels(self):
        returns = _make_returns(100)
        result = detect_volatility_regime(returns)
        assert set(result.unique()).issubset({REGIME_VOL_LOW, REGIME_VOL_MEDIUM, REGIME_VOL_HIGH})

    def test_all_three_regimes_present_on_large_sample(self):
        returns = _make_returns(300)
        result = detect_volatility_regime(returns)
        # With 300 samples and default 33/67 percentiles all three buckets should appear
        assert REGIME_VOL_LOW in result.values
        assert REGIME_VOL_HIGH in result.values

    def test_high_vol_period_classified_correctly(self):
        """A spike in volatility should produce a high-vol label."""
        n = 100
        rng = np.random.default_rng(0)
        returns = pd.Series(rng.normal(0, 0.01, n))
        # Inject a very high-volatility spike at the end
        returns.iloc[-20:] = rng.normal(0, 0.5, 20)
        result = detect_volatility_regime(returns, window=5)
        # The final 20 periods should contain at least some high-vol labels
        assert REGIME_VOL_HIGH in result.iloc[-20:].values

    def test_works_with_short_series(self):
        returns = _make_returns(n=5)
        result = detect_volatility_regime(returns, window=3)
        assert len(result) == 5

    def test_dtype_is_int8(self):
        returns = _make_returns(50)
        result = detect_volatility_regime(returns)
        assert result.dtype == np.int8


# ---------------------------------------------------------------------------
# detect_trend_regime
# ---------------------------------------------------------------------------


class TestDetectTrendRegime:
    def test_returns_series_same_length(self):
        close = _make_close(200)
        result = detect_trend_regime(close)
        assert len(result) == len(close)
        assert result.index.equals(close.index)

    def test_only_valid_labels(self):
        close = _make_close(200)
        result = detect_trend_regime(close)
        assert set(result.unique()).issubset({REGIME_BULL, REGIME_BEAR, REGIME_SIDEWAYS})

    def test_bull_market_predominantly_bull(self):
        """A strongly trending-up series should be labelled mostly bull."""
        n = 200
        close = pd.Series(np.linspace(100, 300, n))
        result = detect_trend_regime(close, fast_ma=20, slow_ma=50)
        bull_frac = (result == REGIME_BULL).mean()
        assert bull_frac > 0.5, f"Expected >50% bull, got {bull_frac:.2%}"

    def test_bear_market_predominantly_bear(self):
        """A strongly trending-down series should be labelled mostly bear."""
        n = 200
        close = pd.Series(np.linspace(300, 100, n))
        result = detect_trend_regime(close, fast_ma=20, slow_ma=50)
        bear_frac = (result == REGIME_BEAR).mean()
        assert bear_frac > 0.5, f"Expected >50% bear, got {bear_frac:.2%}"

    def test_short_series_no_crash(self):
        close = _make_close(n=10)
        result = detect_trend_regime(close, fast_ma=5, slow_ma=10)
        assert len(result) == 10

    def test_dtype_is_int8(self):
        close = _make_close(100)
        result = detect_trend_regime(close)
        assert result.dtype == np.int8


# ---------------------------------------------------------------------------
# detect_hmm_regime
# ---------------------------------------------------------------------------


class TestDetectHmmRegime:
    def test_returns_series_same_length(self):
        returns = _make_returns(200)
        result = detect_gmm_regime(returns)
        assert len(result) == len(returns)
        assert result.index.equals(returns.index)

    def test_labels_within_range_2_states(self):
        returns = _make_returns(200)
        result = detect_gmm_regime(returns, n_states=2)
        assert set(result.unique()).issubset({0, 1})

    def test_labels_within_range_3_states(self):
        returns = _make_returns(200)
        result = detect_gmm_regime(returns, n_states=3)
        assert set(result.unique()).issubset({0, 1, 2})

    def test_fallback_on_tiny_series(self):
        """Too few samples should return all-zero fallback without raising."""
        returns = _make_returns(n=5)
        result = detect_gmm_regime(returns, n_states=2)
        assert len(result) == 5
        assert (result == 0).all()

    def test_n_states_clamped_to_minimum_2(self):
        returns = _make_returns(100)
        result = detect_gmm_regime(returns, n_states=1)
        assert set(result.unique()).issubset({0, 1})

    def test_dtype_is_int8(self):
        returns = _make_returns(100)
        result = detect_gmm_regime(returns)
        assert result.dtype == np.int8

    def test_deterministic_with_fixed_seed(self):
        returns = _make_returns(200, seed=7)
        r1 = detect_gmm_regime(returns, random_state=0)
        r2 = detect_gmm_regime(returns, random_state=0)
        pd.testing.assert_series_equal(r1, r2)


# ---------------------------------------------------------------------------
# classify_regime
# ---------------------------------------------------------------------------


class TestClassifyRegime:
    def test_returns_dataframe_with_expected_columns(self):
        close = _make_close(200)
        returns = close.pct_change(1).fillna(0)
        result = classify_regime(close, returns)
        assert isinstance(result, pd.DataFrame)
        assert COL_REGIME_VOLATILITY in result.columns
        assert COL_REGIME_TREND in result.columns
        assert COL_REGIME_GMM in result.columns

    def test_output_length_matches_input(self):
        close = _make_close(200)
        returns = close.pct_change(1).fillna(0)
        result = classify_regime(close, returns)
        assert len(result) == 200

    def test_index_matches_close(self):
        close = _make_close(200)
        returns = close.pct_change(1).fillna(0)
        result = classify_regime(close, returns)
        assert result.index.equals(close.index)

    def test_all_columns_integer(self):
        close = _make_close(200)
        returns = close.pct_change(1).fillna(0)
        result = classify_regime(close, returns)
        for col in [COL_REGIME_VOLATILITY, COL_REGIME_TREND, COL_REGIME_GMM]:
            assert np.issubdtype(result[col].dtype, np.integer), f"{col} should be integer"

    def test_no_nan_values(self):
        close = _make_close(200)
        returns = close.pct_change(1).fillna(0)
        result = classify_regime(close, returns)
        assert not result.isna().any().any()

    def test_custom_n_hmm_states(self):
        close = _make_close(200)
        returns = close.pct_change(1).fillna(0)
        result = classify_regime(close, returns, n_gmm_states=3)
        assert set(result[COL_REGIME_GMM].unique()).issubset({0, 1, 2})

    def test_works_with_short_series(self):
        close = _make_close(n=15)
        returns = close.pct_change(1).fillna(0)
        result = classify_regime(close, returns)
        assert len(result) == 15


# ---------------------------------------------------------------------------
# Integration: regime features appear in FeatureEngineering output
# ---------------------------------------------------------------------------


class TestRegimeInFeatureEngineering:
    def test_regime_columns_present_in_feature_matrix(self):
        from src.feature.feature_engineering import FeatureEngineering

        ohlcv = make_ohlcv(n=200, seed=42)
        fe = FeatureEngineering(ohlcv)
        fe.create_target_features(target_days=5)
        x, _, _, _, feature_cols = fe.prepare_features()
        for col in [COL_REGIME_VOLATILITY, COL_REGIME_TREND, COL_REGIME_GMM]:
            assert col in x.columns, f"Expected {col!r} in feature matrix"

    def test_regime_columns_have_no_nan_after_prepare(self):
        from src.feature.feature_engineering import FeatureEngineering

        ohlcv = make_ohlcv(n=200, seed=0)
        fe = FeatureEngineering(ohlcv)
        fe.create_target_features(target_days=5)
        x, _, _, _, _ = fe.prepare_features()
        for col in [COL_REGIME_VOLATILITY, COL_REGIME_TREND, COL_REGIME_GMM]:
            assert not x[col].isna().any(), f"{col} should have no NaN"
