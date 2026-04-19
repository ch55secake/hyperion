"""Shared fixtures for regression tests."""

import pytest

from src.feature.feature_engineering import FeatureEngineering
from src.model import XGBoostStockPredictor, LightGBMStockPredictor
from tests.helpers import make_ohlcv, fast_xgb_params, fast_lgb_params


@pytest.fixture(scope="session")
def synthetic_ohlcv():
    """200 rows of deterministic OHLCV data."""
    return make_ohlcv(n=200, seed=42)


@pytest.fixture(scope="session")
def trained_xgb_model(synthetic_ohlcv):
    """Pre-trained XGBoost model on synthetic data; returns (model, x_train, x_test, y_train, y_test)."""
    fe = FeatureEngineering(synthetic_ohlcv)
    fe.create_target_features()
    x, y, *_ = fe.prepare_features()
    split = int(len(x) * 0.8)
    x_train, x_test = x.iloc[:split], x.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = XGBoostStockPredictor(params=fast_xgb_params())
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

    model = LightGBMStockPredictor(params=fast_lgb_params())
    model.train(x_train, y_train)
    return model, x_train, x_test, y_train, y_test
