"""Shared fixtures and helpers for the test suite."""

import numpy as np
import pandas as pd
import pytest

from src.feature.feature_engineering import FeatureEngineering
from src.model import LightGBMStockPredictor, XGBoostStockPredictor
from src.model.model import Model
from tests.helpers import fast_lgb_params, fast_xgb_params, make_ohlcv


class _DummyPredictor(Model):
    """Constant-output predictor used across multiple test modules."""

    def __init__(self, params=None):
        super().__init__("dummy", params=params)
        self._constant = 0.0

    def train(self, x_train, y_train, x_val=None, y_val=None):
        self._prepare_columns(x_train)
        self.model = True
        self._constant = float(np.mean(y_train))
        feature_names = list(x_train.columns)
        self.feature_importance = pd.DataFrame(
            {"feature": feature_names, "importance": np.ones(len(feature_names)) / len(feature_names)}
        )

    def predict(self, x):
        x_processed = self._prepare_prediction(x)
        return np.full(len(x_processed), self._constant)


class _FakeSim:
    """Minimal simulator stub used by strategy unit tests."""

    def __init__(self):
        self.transaction_cost = 0.001
        self.trades = []


@pytest.fixture
def dummy_predictor_cls():
    """Return the _DummyPredictor class so tests can instantiate it freely."""
    return _DummyPredictor


@pytest.fixture
def fake_simulator():
    return _FakeSim()


@pytest.fixture
def synthetic_ohlcv():
    """200 rows of deterministic OHLCV data indexed by calendar day."""
    rng = np.random.default_rng(42)
    n = 200
    prices = np.clip(100 + np.cumsum(rng.normal(0, 1, n)), 10, None)
    return pd.DataFrame(
        {
            "Open": prices * (1 + rng.normal(0, 0.002, n)),
            "High": prices * (1 + np.abs(rng.normal(0, 0.005, n))),
            "Low": prices * (1 - np.abs(rng.normal(0, 0.005, n))),
            "Close": prices,
            "Volume": rng.integers(1_000, 10_000, n).astype(float),
        },
        index=pd.date_range("2022-01-01", periods=n, freq="D"),
    )


def _make_trained_split():
    """Build a train/test split from deterministic synthetic OHLCV data."""
    ohlcv = make_ohlcv(n=200, seed=42)
    fe = FeatureEngineering(ohlcv)
    fe.create_target_features()
    x, y, *_ = fe.prepare_features()
    split = int(len(x) * 0.8)
    return x.iloc[:split], x.iloc[split:], y.iloc[:split], y.iloc[split:]


@pytest.fixture(scope="session")
def trained_xgb_model():
    """Pre-trained XGBoost model on synthetic data; returns (model, x_train, x_test, y_train, y_test)."""
    x_train, x_test, y_train, y_test = _make_trained_split()
    model = XGBoostStockPredictor(params=fast_xgb_params())
    model.train(x_train, y_train)
    return model, x_train, x_test, y_train, y_test


@pytest.fixture(scope="session")
def trained_lgb_model():
    """Pre-trained LightGBM model on synthetic data; returns (model, x_train, x_test, y_train, y_test)."""
    x_train, x_test, y_train, y_test = _make_trained_split()
    model = LightGBMStockPredictor(params=fast_lgb_params())
    model.train(x_train, y_train)
    return model, x_train, x_test, y_train, y_test
