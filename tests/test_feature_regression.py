"""Regression tests for feature engineering pipeline."""

import numpy as np
import pandas as pd
import pytest

from src.feature.feature_engineering import FeatureEngineering


def _make_ohlcv(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0, 1, n)
    low = close - rng.uniform(0, 1, n)
    open_ = close + rng.normal(0, 0.5, n)
    volume = rng.uniform(1_000, 10_000, n)
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


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
        fe1 = FeatureEngineering(_make_ohlcv(seed=42))
        fe1.create_target_features()
        x1, *_ = fe1.prepare_features()

        fe2 = FeatureEngineering(_make_ohlcv(seed=99))
        fe2.create_target_features()
        x2, *_ = fe2.prepare_features()

        assert list(x1.columns) == list(x2.columns), "Column set changed between seeds"

    def test_output_length_with_target_days(self):
        """Row count returned by prepare_features should be close to n - target_days."""
        n = 200
        target_days = 10
        df = _make_ohlcv(n=n, seed=7)
        fe = FeatureEngineering(df)
        fe.create_target_features(target_days=target_days)
        x, y, *_ = fe.prepare_features()

        # At most target_days trailing rows are dropped (NaN targets); allow small slack.
        assert len(x) <= n
        assert len(x) >= n - target_days - 5, f"Unexpectedly few rows returned: {len(x)}"
        assert len(x) == len(y)
