"""Shared helper utilities for the regression test suite."""

import numpy as np
import pandas as pd


def make_ohlcv(n: int = 200, seed: int = 42) -> pd.DataFrame:
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


def fast_xgb_params() -> dict:
    """Return lightweight XGBoost params for use in tests."""
    return {
        "objective": "reg:squarederror",
        "n_estimators": 20,
        "max_depth": 3,
        "tree_method": "hist",
        "seed": 42,
        "enable_categorical": True,
    }


def fast_lgb_params() -> dict:
    """Return lightweight LightGBM params for use in tests."""
    return {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "n_estimators": 20,
        "max_depth": 3,
        "num_leaves": 8,
        "seed": 42,
    }
