"""Unit tests for src/feature/feature_engineering.py"""

import numpy as np
import pandas as pd
import pytest

from src.feature.feature_engineering import FeatureEngineering


def _make_df(n=100, seed=42):
    """Create a minimal OHLCV DataFrame."""
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0, 1, n)
    low = close - rng.uniform(0, 1, n)
    open_ = close + rng.normal(0, 0.5, n)
    volume = rng.uniform(1_000, 10_000, n)
    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        }
    )


class TestFeatureEngineeringInit:
    def test_raises_on_missing_column(self):
        df = pd.DataFrame({"Open": [1.0], "Close": [1.0]})
        with pytest.raises(ValueError, match="Missing required column"):
            FeatureEngineering(df)

    def test_accepts_valid_ohlcv(self):
        df = _make_df()
        fe = FeatureEngineering(df)
        assert fe is not None

    def test_does_not_mutate_original(self):
        df = _make_df()
        original_cols = set(df.columns)
        FeatureEngineering(df)
        # The class should work on an internal copy
        assert set(df.columns) == original_cols


class TestAddAllTechnicalIndicators:
    def test_adds_new_columns(self):
        df = _make_df(n=120)
        fe = FeatureEngineering(df)
        fe.add_all_technical_indicators()
        result = fe.get_df()
        assert len(result.columns) > 5

    def test_is_idempotent(self):
        df = _make_df(n=120)
        fe = FeatureEngineering(df)
        fe.add_all_technical_indicators()
        cols_first = set(fe.get_df().columns)
        fe.add_all_technical_indicators()
        cols_second = set(fe.get_df().columns)
        assert cols_first == cols_second

    def test_no_inf_values(self):
        df = _make_df(n=120)
        fe = FeatureEngineering(df)
        fe.add_all_technical_indicators()
        result = fe.get_df()
        numeric = result.select_dtypes(include="number")
        assert not np.isinf(numeric.values).any()

    def test_sma_columns_present(self):
        df = _make_df(n=120)
        fe = FeatureEngineering(df)
        fe.add_all_technical_indicators()
        result = fe.get_df()
        assert "SMA_5" in result.columns
        assert "SMA_50" in result.columns

    def test_rsi_column_present(self):
        df = _make_df(n=120)
        fe = FeatureEngineering(df)
        fe.add_all_technical_indicators()
        result = fe.get_df()
        assert "RSI" in result.columns

    def test_skips_indicators_when_insufficient_rows(self):
        # With only 10 rows, RSI (requires 14) should not be computed
        df = _make_df(n=10)
        fe = FeatureEngineering(df)
        fe.add_all_technical_indicators()
        result = fe.get_df()
        assert "RSI" not in result.columns


class TestCreateTargetFeatures:
    def test_target_column_added(self):
        df = _make_df(n=120)
        fe = FeatureEngineering(df)
        result = fe.create_target_features(target_days=5)
        assert "Target" in result.columns

    def test_target_length_unchanged(self):
        df = _make_df(n=120)
        fe = FeatureEngineering(df)
        result = fe.create_target_features(target_days=5)
        assert len(result) == 120


class TestPrepareFeatures:
    def test_returns_tuple_of_five(self):
        df = _make_df(n=120)
        fe = FeatureEngineering(df)
        fe.create_target_features(target_days=5)
        x, y, *_ = fe.prepare_features()
        assert isinstance(x, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(x) == len(y)

    def test_no_nan_in_features(self):
        df = _make_df(n=120)
        fe = FeatureEngineering(df)
        fe.create_target_features(target_days=5)
        x, _, _, _, _ = fe.prepare_features()
        assert not x.isna().any().any()

    def test_is_cached(self):
        df = _make_df(n=120)
        fe = FeatureEngineering(df)
        fe.create_target_features(target_days=5)
        result1 = fe.prepare_features()
        result2 = fe.prepare_features()
        assert result1 is result2

    def test_raises_when_no_valid_data(self):
        # Provide a minimal DataFrame that will have no valid Target rows
        df = pd.DataFrame(
            {
                "Open": [10.0, 11.0],
                "High": [11.0, 12.0],
                "Low": [9.0, 10.0],
                "Close": [10.5, 11.5],
                "Volume": [1000.0, 1100.0],
            }
        )
        fe = FeatureEngineering(df)
        fe.create_target_features(target_days=5)
        with pytest.raises(ValueError, match="No valid data"):
            fe.prepare_features()
