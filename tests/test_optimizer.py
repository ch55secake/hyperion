"""Unit tests for src/optimise/optimizer.py (StockModelOptimizer)."""

import numpy as np
import pandas as pd
import pytest
import xgboost as _xgb_module

from src.optimise.optimizer import StockModelOptimizer

# XGBoost 3.x moved early_stopping_rounds from fit() to the constructor.
# The project pins xgboost==2.0.3 where the optimizer was written.
_XGB_MAJOR = int(_xgb_module.__version__.split(".")[0])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_numeric_data(n_train=60, n_val=20, n_features=5, seed=0):
    rng = np.random.default_rng(seed)
    cols = [f"f{i}" for i in range(n_features)]
    x_train = pd.DataFrame(rng.normal(0, 1, (n_train, n_features)), columns=cols)
    x_val = pd.DataFrame(rng.normal(0, 1, (n_val, n_features)), columns=cols)
    y_train = pd.Series(rng.normal(0, 0.02, n_train))
    y_val = pd.Series(rng.normal(0, 0.02, n_val))
    return x_train, y_train, x_val, y_val


def _make_data_with_categoricals(n_train=60, n_val=20, seed=0):
    rng = np.random.default_rng(seed)
    x_train = pd.DataFrame(
        {
            "num_a": rng.normal(0, 1, n_train),
            "num_b": rng.normal(0, 1, n_train),
            "sector": pd.Categorical(rng.choice(["tech", "finance", "health"], n_train)),
        }
    )
    x_val = pd.DataFrame(
        {
            "num_a": rng.normal(0, 1, n_val),
            "num_b": rng.normal(0, 1, n_val),
            "sector": pd.Categorical(rng.choice(["tech", "finance", "health"], n_val)),
        }
    )
    y_train = pd.Series(rng.normal(0, 0.02, n_train))
    y_val = pd.Series(rng.normal(0, 0.02, n_val))
    return x_train, y_train, x_val, y_val


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestStockModelOptimizerInit:
    def test_numeric_columns_detected(self):
        x_train, y_train, x_val, y_val = _make_numeric_data()
        opt = StockModelOptimizer(x_train, y_train, x_val, y_val, n_trials=1)
        assert len(opt.numeric_columns) == 5

    def test_categorical_columns_empty_for_numeric_only_data(self):
        x_train, y_train, x_val, y_val = _make_numeric_data()
        opt = StockModelOptimizer(x_train, y_train, x_val, y_val, n_trials=1)
        assert opt.categorical_columns == []

    def test_categorical_columns_detected(self):
        x_train, y_train, x_val, y_val = _make_data_with_categoricals()
        opt = StockModelOptimizer(x_train, y_train, x_val, y_val, n_trials=1)
        assert "sector" in opt.categorical_columns

    def test_numeric_columns_detected_with_categoricals(self):
        x_train, y_train, x_val, y_val = _make_data_with_categoricals()
        opt = StockModelOptimizer(x_train, y_train, x_val, y_val, n_trials=1)
        assert "num_a" in opt.numeric_columns
        assert "num_b" in opt.numeric_columns

    def test_best_params_start_as_none(self):
        x_train, y_train, x_val, y_val = _make_numeric_data()
        opt = StockModelOptimizer(x_train, y_train, x_val, y_val, n_trials=1)
        assert opt.best_xgb_params is None
        assert opt.best_lgb_params is None

    def test_studies_start_as_none(self):
        x_train, y_train, x_val, y_val = _make_numeric_data()
        opt = StockModelOptimizer(x_train, y_train, x_val, y_val, n_trials=1)
        assert opt.xgb_study is None
        assert opt.lgb_study is None

    def test_random_state_stored(self):
        x_train, y_train, x_val, y_val = _make_numeric_data()
        opt = StockModelOptimizer(x_train, y_train, x_val, y_val, n_trials=1, random_state=123)
        assert opt.random_state == 123


# ---------------------------------------------------------------------------
# _process_features
# ---------------------------------------------------------------------------


class TestProcessFeatures:
    def test_object_columns_converted_to_category(self):
        x_train, y_train, x_val, y_val = _make_numeric_data()
        opt = StockModelOptimizer(x_train, y_train, x_val, y_val, n_trials=1)
        df = pd.DataFrame({"a": [1.0, 2.0], "label": ["x", "y"]})
        processed = opt._process_features(df)
        assert processed["label"].dtype.name == "category"

    def test_numeric_columns_unchanged(self):
        x_train, y_train, x_val, y_val = _make_numeric_data()
        opt = StockModelOptimizer(x_train, y_train, x_val, y_val, n_trials=1)
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        processed = opt._process_features(df)
        assert processed["a"].dtype in (np.float64, np.float32, float)

    def test_no_object_columns_returns_unchanged_shape(self):
        x_train, y_train, x_val, y_val = _make_numeric_data()
        opt = StockModelOptimizer(x_train, y_train, x_val, y_val, n_trials=1)
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        processed = opt._process_features(df)
        assert processed.shape == df.shape


# ---------------------------------------------------------------------------
# xgboost_objective (via mock trial)
# ---------------------------------------------------------------------------


class _MockTrial:
    """Minimal Optuna trial stub that returns low-complexity fixed values."""

    def __init__(self):
        self._attrs = {}

    def suggest_float(self, _name, low, _high, log=False):  # noqa: ARG002
        return low

    def suggest_int(self, _name, low, _high):  # noqa: ARG002
        return low

    def set_user_attr(self, key, value):
        self._attrs[key] = value


class TestXGBoostObjective:
    @pytest.mark.skipif(_XGB_MAJOR >= 3, reason="XGBoost >=3 changed early_stopping_rounds API")
    def test_returns_float(self):
        x_train, y_train, x_val, y_val = _make_numeric_data(n_train=80, n_val=20, n_features=3)
        opt = StockModelOptimizer(x_train, y_train, x_val, y_val, n_trials=1)
        trial = _MockTrial()
        result = opt.xgboost_objective(trial)
        assert isinstance(result, float)

    @pytest.mark.skipif(_XGB_MAJOR >= 3, reason="XGBoost >=3 changed early_stopping_rounds API")
    def test_returns_non_negative_rmse(self):
        x_train, y_train, x_val, y_val = _make_numeric_data(n_train=80, n_val=20, n_features=3)
        opt = StockModelOptimizer(x_train, y_train, x_val, y_val, n_trials=1)
        trial = _MockTrial()
        result = opt.xgboost_objective(trial)
        assert result >= 0.0

    @pytest.mark.skipif(_XGB_MAJOR >= 3, reason="XGBoost >=3 changed early_stopping_rounds API")
    def test_sets_user_attributes(self):
        x_train, y_train, x_val, y_val = _make_numeric_data(n_train=80, n_val=20, n_features=3)
        opt = StockModelOptimizer(x_train, y_train, x_val, y_val, n_trials=1)
        trial = _MockTrial()
        opt.xgboost_objective(trial)
        assert "mae" in trial._attrs
        assert "r2" in trial._attrs
        assert "directional_accuracy" in trial._attrs


# ---------------------------------------------------------------------------
# lightgbm_objective (via mock trial)
# ---------------------------------------------------------------------------


class TestLightGBMObjective:
    def test_returns_float(self):
        x_train, y_train, x_val, y_val = _make_numeric_data(n_train=80, n_val=20, n_features=3)
        opt = StockModelOptimizer(x_train, y_train, x_val, y_val, n_trials=1)
        trial = _MockTrial()
        result = opt.lightgbm_objective(trial)
        assert isinstance(result, float)

    def test_returns_non_negative_rmse(self):
        x_train, y_train, x_val, y_val = _make_numeric_data(n_train=80, n_val=20, n_features=3)
        opt = StockModelOptimizer(x_train, y_train, x_val, y_val, n_trials=1)
        trial = _MockTrial()
        result = opt.lightgbm_objective(trial)
        assert result >= 0.0

    def test_sets_user_attributes(self):
        x_train, y_train, x_val, y_val = _make_numeric_data(n_train=80, n_val=20, n_features=3)
        opt = StockModelOptimizer(x_train, y_train, x_val, y_val, n_trials=1)
        trial = _MockTrial()
        opt.lightgbm_objective(trial)
        assert "mae" in trial._attrs
        assert "r2" in trial._attrs
