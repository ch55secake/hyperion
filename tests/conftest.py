"""Shared fixtures for regression tests."""

import numpy as np
import pandas as pd
import pytest

from src.feature.feature_engineering import FeatureEngineering
from src.model import XGBoostStockPredictor, LightGBMStockPredictor


def _make_ohlcv(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Return a deterministic OHLCV DataFrame with a DatetimeIndex."""
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


def _fast_xgb_params() -> dict:
    return {
        "objective": "reg:squarederror",
        "n_estimators": 20,
        "max_depth": 3,
        "tree_method": "hist",
        "seed": 42,
        "enable_categorical": True,
    }


def _fast_lgb_params() -> dict:
    return {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "n_estimators": 20,
        "max_depth": 3,
        "num_leaves": 8,
        "seed": 42,
    }


@pytest.fixture(scope="session")
def synthetic_ohlcv() -> pd.DataFrame:
    """200 rows of deterministic OHLCV data."""
    return _make_ohlcv(n=200, seed=42)


@pytest.fixture(scope="session")
def trained_xgb_model(synthetic_ohlcv):
    """Pre-trained XGBoost model on synthetic data; returns (model, x_train, x_test, y_train, y_test)."""
    fe = FeatureEngineering(synthetic_ohlcv)
    fe.create_target_features()
    x, y, *_ = fe.prepare_features()
    split = int(len(x) * 0.8)
    x_train, x_test = x.iloc[:split], x.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = XGBoostStockPredictor(params=_fast_xgb_params())
    model.train(x_train, y_train)
    return model, x_train, x_test, y_train, y_test


@pytest.fixture(scope="session")
def trained_lgb_model(synthetic_ohlcv):
    """Pre-trained LightGBM model on synthetic data; returns (model, x_train, x_test, y_train, y_test)."""
    fe = FeatureEngineering(synthetic_ohlcv)
    fe.create_target_features()
    x, y, *_ = fe.prepare_features()
    split = int(len(x) * 0.8)
    x_train, x_test = x.iloc[:split], x.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = LightGBMStockPredictor(params=_fast_lgb_params())
    model.train(x_train, y_train)
    return model, x_train, x_test, y_train, y_test
