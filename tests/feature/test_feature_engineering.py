"""Unit tests for src/feature/feature_engineering.py"""

import numpy as np
import pandas as pd
import pytest

from src.feature.feature_engineering import FeatureEngineering
from tests.helpers import make_ohlcv


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


def _make_df_with_categoricals(n=120, seed=42):
    """Create a minimal OHLCV DataFrame that also has ticker/sector/industry columns."""
    df = _make_df(n=n, seed=seed)
    df["ticker"] = "AAPL"
    df["sector"] = "Technology"
    df["industry"] = "Software"
    return df


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


def make_constant_price_df(n=120, price=100.0, volume=5000.0):
    """OHLCV DataFrame where every price is identical (no movement at all)."""
    return pd.DataFrame(
        {
            "Open": [price] * n,
            "High": [price] * n,
            "Low": [price] * n,
            "Close": [price] * n,
            "Volume": [volume] * n,
        }
    )


class TestKnownInputSanity:
    """Value-level checks for analytically predictable outputs on constant-price input."""

    def test_sma_ema_wma_equal_constant_price(self):
        """All moving average columns must equal the constant price when price never changes."""
        price = 100.0
        df = make_constant_price_df(price=price)
        fe = FeatureEngineering(df)
        fe.add_all_technical_indicators()
        result = fe.get_df()

        ma_cols = [
            c for c in result.columns
            if c.startswith(("SMA_", "EMA_", "WMA_", "HMA_")) and c.split("_")[-1].isdigit()
        ]
        assert len(ma_cols) > 0, "Expected at least one moving-average column"
        for col in ma_cols:
            np.testing.assert_allclose(
                result[col].to_numpy(),
                price,
                rtol=1e-10,
                err_msg=f"{col} should equal {price} for constant input",
            )

    def test_atr_zero_for_constant_price(self):
        """ATR must be 0 when High == Low == Close == constant."""
        df = make_constant_price_df()
        fe = FeatureEngineering(df)
        fe.add_all_technical_indicators()
        result = fe.get_df()

        assert "ATR" in result.columns, "ATR column should be present"
        assert (result["ATR"] == 0.0).all(), f"ATR should be 0 for constant price, got: {result['ATR'].unique()}"

    def test_momentum_zero_for_constant_price(self):
        """All Momentum_n columns must be 0 when price never changes."""
        df = make_constant_price_df()
        fe = FeatureEngineering(df)
        fe.add_all_technical_indicators()
        result = fe.get_df()

        momentum_cols = [c for c in result.columns if c.startswith("Momentum_") and "_Ratio_" not in c]
        assert len(momentum_cols) > 0, "Expected at least one Momentum column"
        for col in momentum_cols:
            assert (result[col] == 0.0).all(), (
                f"{col} should be 0 for constant price, got: {result[col].unique()}"
            )


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


class TestMultiHorizonTargets:
    def test_horizon_columns_added(self):
        df = _make_df(n=120)
        fe = FeatureEngineering(df)
        result = fe.create_target_features(target_days=10, horizons=[1, 5, 10, 20])
        for h in [1, 5, 10, 20]:
            assert f"Target_{h}d" in result.columns

    def test_horizon_columns_not_added_by_default(self):
        df = _make_df(n=120)
        fe = FeatureEngineering(df)
        result = fe.create_target_features(target_days=10)
        assert "Target_1d" not in result.columns
        assert "Target_5d" not in result.columns

    def test_horizon_length_unchanged(self):
        df = _make_df(n=120)
        fe = FeatureEngineering(df)
        result = fe.create_target_features(target_days=10, horizons=[1, 5])
        assert len(result) == 120

    def test_horizon_targets_not_in_feature_matrix(self):
        df = _make_df(n=120)
        fe = FeatureEngineering(df)
        fe.create_target_features(target_days=10, horizons=[1, 5, 10, 20])
        x, _, _, _, _ = fe.prepare_features()
        for h in [1, 5, 10, 20]:
            assert f"Target_{h}d" not in x.columns


class TestRiskAdjustedTargets:
    def test_sharpe_columns_added(self):
        df = _make_df(n=120)
        fe = FeatureEngineering(df)
        result = fe.create_target_features(target_days=10, horizons=[5, 10], risk_adjusted=True)
        assert "Target_Sharpe_5d" in result.columns
        assert "Target_Sharpe_10d" in result.columns

    def test_maxdd_columns_added(self):
        df = _make_df(n=120)
        fe = FeatureEngineering(df)
        result = fe.create_target_features(target_days=10, horizons=[5], risk_adjusted=True)
        assert "Target_MaxDD_5d" in result.columns

    def test_sortino_columns_added(self):
        df = _make_df(n=120)
        fe = FeatureEngineering(df)
        result = fe.create_target_features(target_days=10, horizons=[5], risk_adjusted=True)
        assert "Target_Sortino_5d" in result.columns

    def test_risk_adjusted_fallback_to_target_days(self):
        """When horizons=None, risk_adjusted targets use [target_days]."""
        df = _make_df(n=120)
        fe = FeatureEngineering(df)
        result = fe.create_target_features(target_days=5, risk_adjusted=True)
        assert "Target_Sharpe_5d" in result.columns
        assert "Target_MaxDD_5d" in result.columns
        assert "Target_Sortino_5d" in result.columns

    def test_risk_adjusted_not_in_feature_matrix(self):
        df = _make_df(n=120)
        fe = FeatureEngineering(df)
        fe.create_target_features(target_days=10, horizons=[5], risk_adjusted=True)
        x, _, _, _, _ = fe.prepare_features()
        assert "Target_Sharpe_5d" not in x.columns
        assert "Target_MaxDD_5d" not in x.columns
        assert "Target_Sortino_5d" not in x.columns

    def test_maxdd_target_is_less_than_or_equal_to_raw_return(self):
        """MDD-adjusted return should be <= the raw forward return when return is positive."""
        df = _make_df(n=120)
        fe = FeatureEngineering(df)
        result = fe.create_target_features(target_days=5, horizons=[5], risk_adjusted=True)
        raw = result["Target_5d"].dropna()
        mdd_adj = result["Target_MaxDD_5d"].dropna()
        positive_mask = (raw > 0) & (mdd_adj.index.isin(raw.index))
        if positive_mask.any():
            assert (mdd_adj[positive_mask] <= raw[positive_mask] + 1e-9).all()


class TestClassificationTargets:
    def test_binary_column_added(self):
        df = _make_df(n=120)
        fe = FeatureEngineering(df)
        result = fe.create_target_features(target_days=5, classification=True)
        assert "Target_Binary" in result.columns

    def test_ternary_column_added(self):
        df = _make_df(n=120)
        fe = FeatureEngineering(df)
        result = fe.create_target_features(target_days=5, classification=True)
        assert "Target_Ternary" in result.columns

    def test_binary_values_are_zero_or_one(self):
        df = _make_df(n=120)
        fe = FeatureEngineering(df)
        result = fe.create_target_features(target_days=5, classification=True)
        valid = result["Target_Binary"].dropna()
        assert set(valid.unique()).issubset({0, 1})

    def test_ternary_values_are_minus_one_zero_or_one(self):
        df = _make_df(n=120)
        fe = FeatureEngineering(df)
        result = fe.create_target_features(target_days=5, classification=True)
        valid = result["Target_Ternary"].dropna()
        assert set(valid.unique()).issubset({-1, 0, 1})

    def test_classification_not_in_feature_matrix(self):
        df = _make_df(n=120)
        fe = FeatureEngineering(df)
        fe.create_target_features(target_days=5, classification=True)
        x, _, _, _, _ = fe.prepare_features()
        assert "Target_Binary" not in x.columns
        assert "Target_Ternary" not in x.columns

    def test_invalid_thresholds_raise(self):
        df = _make_df(n=120)
        fe = FeatureEngineering(df)
        with pytest.raises(ValueError, match="down_threshold"):
            fe.create_target_features(target_days=5, classification=True, up_threshold=-0.05, down_threshold=0.05)

    def test_custom_thresholds_respected(self):
        """All returns above 5% should be labelled 1 with up_threshold=0.05."""
        df = _make_df(n=120)
        fe = FeatureEngineering(df)
        result = fe.create_target_features(target_days=5, classification=True, up_threshold=0.05, down_threshold=-0.05)
        forward_return = result["Close"].pct_change(5).shift(-5)
        binary = result["Target_Binary"]
        mask = forward_return > 0.05
        assert (binary[mask] == 1).all()


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


class TestPrepareFeaturesWithScale:
    def test_scale_preserves_categorical_columns(self):
        df = _make_df_with_categoricals(n=120)
        fe = FeatureEngineering(df)
        fe.create_target_features(target_days=5)
        x, _, _, _, _ = fe.prepare_features(scale=True)
        assert "ticker" in x.columns
        assert "sector" in x.columns
        assert "industry" in x.columns

    def test_scale_numeric_columns_are_scaled(self):
        df = _make_df_with_categoricals(n=120)
        fe = FeatureEngineering(df)
        fe.create_target_features(target_days=5)
        x, _, _, _, _ = fe.prepare_features(scale=True)
        numeric_cols = x.select_dtypes(include="number").columns
        assert len(numeric_cols) > 0
        for col in numeric_cols:
            assert abs(x[col].mean()) < 1.0

    def test_scale_categorical_values_unchanged(self):
        df = _make_df_with_categoricals(n=120)
        fe = FeatureEngineering(df)
        fe.create_target_features(target_days=5)
        x, _, _, _, _ = fe.prepare_features(scale=True)
        assert (x["ticker"] == "AAPL").all()
        assert (x["sector"] == "Technology").all()
        assert (x["industry"] == "Software").all()

    def test_scale_same_row_count(self):
        df = _make_df_with_categoricals(n=120)
        fe_unscaled = FeatureEngineering(df.copy())
        fe_unscaled.create_target_features(target_days=5)
        x_unscaled, y_unscaled, _, _, _ = fe_unscaled.prepare_features(scale=False)

        fe_scaled = FeatureEngineering(df.copy())
        fe_scaled.create_target_features(target_days=5)
        x_scaled, y_scaled, _, _, _ = fe_scaled.prepare_features(scale=True)

        assert len(x_scaled) == len(x_unscaled)
        assert len(y_scaled) == len(y_unscaled)


class TestTemporalOrdering:
    def test_shuffled_input_is_sorted_chronologically(self):
        """FeatureEngineering must sort data by date even when input is shuffled."""
        ohlcv = make_ohlcv(n=120, seed=0)
        shuffled = ohlcv.sample(frac=1, random_state=99)
        # Sorting happens in __init__ via _ensure_required_columns; check immediately.
        fe = FeatureEngineering(shuffled)
        assert fe.get_df().index.is_monotonic_increasing

    def test_prepare_features_index_is_sorted(self):
        """prepare_features must return x/y/dates in chronological order."""
        ohlcv = make_ohlcv(n=120, seed=1)
        shuffled = ohlcv.sample(frac=1, random_state=7)
        fe = FeatureEngineering(shuffled)
        fe.create_target_features(target_days=5)
        x, y, dates, prices, _ = fe.prepare_features()
        assert list(dates) == sorted(dates)

    def test_positional_split_is_temporal(self):
        """A positional 80/20 split on sorted data must have all train dates before test dates."""
        ohlcv = make_ohlcv(n=200, seed=42)
        fe = FeatureEngineering(ohlcv)
        fe.create_target_features(target_days=5)
        x, y, dates, prices, _ = fe.prepare_features()
        split = int(len(x) * 0.8)
        train_dates = dates[:split]
        test_dates = dates[split:]
        assert max(train_dates) < min(test_dates)

    def test_positional_split_is_temporal_on_shuffled_input(self):
        """An 80/20 split must still be temporal even if input data was shuffled."""
        ohlcv = make_ohlcv(n=200, seed=42)
        shuffled = ohlcv.sample(frac=1, random_state=13)
        fe = FeatureEngineering(shuffled)
        fe.create_target_features(target_days=5)
        x, y, dates, prices, _ = fe.prepare_features()
        split = int(len(x) * 0.8)
        assert split > 0 and split < len(x)
        assert max(dates[:split]) < min(dates[split:])
