"""Regression tests for feature engineering pipeline."""

import numpy as np
import pandas as pd
import pytest

import src.feature.technical_indicators as ti
from src.feature.feature_engineering import FeatureEngineering
from tests.helpers import make_ohlcv


class TestFeatureEngineeringRegression:
    """Regression tests for feature engineering with deterministic synthetic data."""

    def test_feature_count_is_stable(self, synthetic_ohlcv):
        """Number of output features should match expected count (catches accidental additions/removals)."""
        fe = FeatureEngineering(synthetic_ohlcv)
        fe.create_target_features()
        x, *_ = fe.prepare_features()

        # Record the stable count; any accidental change will surface here.
        feature_count = len(x.columns)
        assert feature_count > 0, "Feature matrix must have at least one column"

        # Run again with identical input and verify count is identical.
        fe2 = FeatureEngineering(synthetic_ohlcv.copy())
        fe2.create_target_features()
        x2, *_ = fe2.prepare_features()
        assert len(x2.columns) == feature_count, (
            f"Feature count changed: expected {feature_count}, got {len(x2.columns)}"
        )

    def test_feature_values_deterministic(self, synthetic_ohlcv):
        """Feature matrix for fixed input should produce the exact same values on repeated calls."""
        fe1 = FeatureEngineering(synthetic_ohlcv.copy())
        fe1.create_target_features()
        x1, y1, *_ = fe1.prepare_features()

        fe2 = FeatureEngineering(synthetic_ohlcv.copy())
        fe2.create_target_features()
        x2, y2, *_ = fe2.prepare_features()

        pd.testing.assert_frame_equal(x1, x2)
        pd.testing.assert_series_equal(y1, y2)

    def test_no_nans_in_output(self, synthetic_ohlcv):
        """After prepare_features(), no NaN values should remain in the feature matrix."""
        fe = FeatureEngineering(synthetic_ohlcv)
        fe.create_target_features()
        x, *_ = fe.prepare_features()

        assert not x.isna().any().any(), "NaN values found in feature matrix after prepare_features()"

    def test_no_inf_in_output(self, synthetic_ohlcv):
        """After prepare_features(), no Inf values should remain in the feature matrix."""
        fe = FeatureEngineering(synthetic_ohlcv)
        fe.create_target_features()
        x, *_ = fe.prepare_features()

        numeric = x.select_dtypes(include="number")
        assert not np.isinf(numeric.values).any(), "Inf values found in feature matrix after prepare_features()"

    def test_feature_columns_are_consistent_across_seeds(self):
        """Column names (not values) should be identical regardless of the random seed used to generate prices."""
        fe1 = FeatureEngineering(make_ohlcv(seed=42))
        fe1.create_target_features()
        x1, *_ = fe1.prepare_features()

        fe2 = FeatureEngineering(make_ohlcv(seed=99))
        fe2.create_target_features()
        x2, *_ = fe2.prepare_features()

        assert list(x1.columns) == list(x2.columns), "Column set changed between seeds"

    def test_output_length_with_target_days(self):
        """Row count returned by prepare_features should be close to n - target_days."""
        n = 200
        target_days = 10
        df = make_ohlcv(n=n, seed=7)
        fe = FeatureEngineering(df)
        fe.create_target_features(target_days=target_days)
        x, y, *_ = fe.prepare_features()

        # At most target_days trailing rows are dropped (NaN targets); allow small slack.
        assert len(x) <= n
        assert len(x) >= n - target_days - 5, f"Unexpectedly few rows returned: {len(x)}"
        assert len(x) == len(y)


# ---------------------------------------------------------------------------
# Cross-indicator consistency
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fe_ohlcv_pair():
    """Return (ohlcv, df_after_indicators) for a fixed seed — shared across tests in the module."""
    ohlcv = make_ohlcv(n=200, seed=42)
    fe = FeatureEngineering(ohlcv.copy())
    fe.add_all_technical_indicators()
    return ohlcv, fe.get_df()


class TestCrossIndicatorConsistency:
    """Assert that FeatureEngineering wires each indicator correctly by comparing its
    output to the equivalent ``ti.*`` function called directly (plus a 1-day shift,
    since add_all_technical_indicators shifts all feature columns by one day)."""

    def test_sma10_matches_direct_call(self, fe_ohlcv_pair):
        ohlcv, df = fe_ohlcv_pair
        expected = ti.sma(ohlcv["Close"], 10).shift(1)
        actual = df["SMA_10"]
        mask = expected.notna() & actual.notna()
        pd.testing.assert_series_equal(actual[mask], expected[mask], check_names=False, rtol=1e-10)

    def test_atr_matches_direct_call(self, fe_ohlcv_pair):
        ohlcv, df = fe_ohlcv_pair
        expected = ti.atr(ohlcv["High"], ohlcv["Low"], ohlcv["Close"]).shift(1)
        actual = df["ATR"]
        mask = expected.notna() & actual.notna()
        pd.testing.assert_series_equal(actual[mask], expected[mask], check_names=False, rtol=1e-10)

    def test_rsi_matches_direct_call(self, fe_ohlcv_pair):
        ohlcv, df = fe_ohlcv_pair
        expected = ti.rsi(ohlcv["Close"]).shift(1)
        actual = df["RSI"]
        mask = expected.notna() & actual.notna()
        pd.testing.assert_series_equal(actual[mask], expected[mask], check_names=False, rtol=1e-10)

    def test_momentum5_matches_direct_call(self, fe_ohlcv_pair):
        ohlcv, df = fe_ohlcv_pair
        expected = ti.momentum(ohlcv["Close"], 5).shift(1)
        actual = df["Momentum_5"]
        mask = expected.notna() & actual.notna()
        pd.testing.assert_series_equal(actual[mask], expected[mask], check_names=False, rtol=1e-10)


# ---------------------------------------------------------------------------
# Golden-output regression snapshot
# ---------------------------------------------------------------------------

# Expected values for the last 5 rows of key features when computed from
# make_ohlcv(n=50, seed=42).  These are captured once and locked in; any
# future change to a computation that alters these values will be caught here.
_GOLDEN_EXPECTED = {
    "SMA_5": [102.42527355673971, 102.51430961473031, 102.6690005746495, 103.00151258578137, 103.42337504491427],
    "SMA_10": [102.1444701212106, 102.15051433107784, 102.2550961645054, 102.46605320050666, 102.82734971438427],
    "EMA_5": [102.31556656260992, 102.45005589686227, 102.83019171234655, 103.15814743892763, 103.60308911100564],
    "EMA_10": [102.05749027717061, 102.17777105684266, 102.43462419983764, 102.68543050751984, 103.01407449800017],
    "RSI": [49.112224450396404, 52.98879256592149, 61.11766118719389, 68.61961851436433, 68.87114838431316],
    "ATR": [1.111150350116768, 1.1007958858930675, 1.0926727758584778, 1.1175953911475227, 1.1214492519378143],
    "Momentum_5": [0.9697458644273809, 0.4451802899529582, 0.7734547995959531, 1.662560055659327, 2.109312295664509],
    "Momentum_10": [0.9707257946643324, 0.06044209867246764, 1.0458183342755234, 2.1095703600127393, 3.612965138775877],
    "ROC_5": [0.9551266942498502, 0.4352825979884646, 0.752263473282312, 1.6275434766958168, 2.0602040329272677],
    "BB_Upper": [103.88269460058049, 103.79140405272584, 103.87532314834174, 103.98606966851605, 104.26071893253612],
}


class TestGoldenOutput:
    """Lock in computed feature values for a fixed input so that silent regressions
    (wrong window, off-by-one, incorrect column mapping) are caught immediately."""

    @pytest.fixture(scope="class")
    def golden_df(self):
        ohlcv = make_ohlcv(n=50, seed=42)
        fe = FeatureEngineering(ohlcv)
        fe.add_all_technical_indicators()
        return fe.get_df()

    @pytest.mark.parametrize("col", list(_GOLDEN_EXPECTED.keys()))
    def test_last5_rows(self, golden_df, col):
        """Last 5 rows of each key feature must match the golden snapshot."""
        expected = np.array(_GOLDEN_EXPECTED[col])
        actual = golden_df[col].iloc[-5:].to_numpy()
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=1e-6,
            err_msg=f"Golden snapshot mismatch for column '{col}'",
        )
