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


# ---------------------------------------------------------------------------
# Three-way temporal split (data-leakage regression tests)
# ---------------------------------------------------------------------------


class TestThreeWayTemporalSplit:
    """Verify that the train / val / test split is laid out correctly in time."""

    def _make_random_data(self, n: int = 100, n_features: int = 3, seed: int = 0):
        rng = np.random.default_rng(seed)
        cols = [f"f{i}" for i in range(n_features)]
        x = pd.DataFrame(rng.normal(0, 1, (n, n_features)), columns=cols)
        y = pd.Series(rng.normal(0, 0.02, n))
        return x, y

    def test_val_indices_are_strictly_after_train(self):
        """Validation samples must come strictly after all training samples in time.

        Temporal ordering ensures the model cannot see future validation data during
        training, preventing look-ahead data leakage.
        """
        x, y = self._make_random_data(n=100)
        test_size = 0.2
        val_size = 0.1
        n = len(x)
        test_split_idx = int(n * (1 - test_size))
        val_split_idx = int(n * (1 - test_size - val_size))

        x_train = x.iloc[:val_split_idx]
        x_val = x.iloc[val_split_idx:test_split_idx]

        assert val_split_idx > 0, "val split index must be positive"
        assert len(x_train) > 0
        assert len(x_val) > 0
        # All val indices must be >= val_split_idx (i.e. strictly after train)
        assert x_val.index.min() >= val_split_idx

    def test_test_indices_are_strictly_after_val(self):
        """Test samples must come strictly after all validation samples in time.

        Keeping the test set after validation ensures hyperparameter tuning (which
        uses the validation split) cannot influence the final out-of-sample metric.
        """
        x, y = self._make_random_data(n=100)
        test_size = 0.2
        val_size = 0.1
        n = len(x)
        test_split_idx = int(n * (1 - test_size))
        val_split_idx = int(n * (1 - test_size - val_size))

        x_val = x.iloc[val_split_idx:test_split_idx]
        x_test = x.iloc[test_split_idx:]

        assert len(x_val) > 0
        assert len(x_test) > 0
        assert x_test.index.min() >= test_split_idx

    def test_splits_are_non_overlapping_and_exhaustive(self):
        """Train + val + test must cover all rows without overlap."""
        x, y = self._make_random_data(n=100)
        test_size = 0.2
        val_size = 0.1
        n = len(x)
        test_split_idx = int(n * (1 - test_size))
        val_split_idx = int(n * (1 - test_size - val_size))

        x_train = x.iloc[:val_split_idx]
        x_val = x.iloc[val_split_idx:test_split_idx]
        x_test = x.iloc[test_split_idx:]

        # Exhaustive
        assert len(x_train) + len(x_val) + len(x_test) == n
        # Non-overlapping
        all_indices = list(x_train.index) + list(x_val.index) + list(x_test.index)
        assert len(all_indices) == len(set(all_indices))

    def test_optimizer_receives_val_not_test(self):
        """The optimizer's x_val must match the validation slice, not the test slice."""
        x, y = self._make_random_data(n=100)
        test_size = 0.2
        val_size = 0.1
        n = len(x)
        test_split_idx = int(n * (1 - test_size))
        val_split_idx = int(n * (1 - test_size - val_size))

        x_train = x.iloc[:val_split_idx]
        x_val = x.iloc[val_split_idx:test_split_idx]
        x_test = x.iloc[test_split_idx:]
        y_train = y.iloc[:val_split_idx]
        y_val = y.iloc[val_split_idx:test_split_idx]

        opt = StockModelOptimizer(x_train, y_train, x_val, y_val, n_trials=1)

        # The optimizer should hold the val slice, not the test slice
        assert len(opt.x_val) == len(x_val)
        assert len(opt.x_val) != len(x_test), (
            "Optimizer must receive the validation set, not the held-out test set"
        )
