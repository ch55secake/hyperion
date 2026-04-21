"""Unit tests for src/model/stacker/weighted_ensemble.py (StackedStockPredictor)."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.model.model import Model
from src.model.stacker.weighted_ensemble import StackedStockPredictor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _ConstantPredictor(Model):
    """Trains to the mean of y_train and returns that constant for every row."""

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


def _make_feature_df(n=50, n_features=4, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {f"f{i}": rng.normal(0, 1, n) for i in range(n_features)},
    )


def _make_target(n=50, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0, 0.02, n))


def _make_stacker(n_train=80, n_val=20):
    """Return a StackedStockPredictor with two ConstantPredictor models."""
    models = {
        "1d": _ConstantPredictor(),
        "1h": _ConstantPredictor(),
    }
    return StackedStockPredictor(models)


def _make_train_data(n_train=80, n_val=20):
    x_train = _make_feature_df(n_train)
    x_val = _make_feature_df(n_val, seed=1)
    y_train = _make_target(n_train)
    y_val = _make_target(n_val, seed=1)
    return {
        "1d": (x_train, y_train, x_val, y_val),
        "1h": (x_train, y_train, x_val, y_val),
    }


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestStackedStockPredictorInit:
    def test_default_weights_are_one(self):
        stacker = _make_stacker()
        assert all(w == 1.0 for w in stacker.weights.values())

    def test_custom_weights_stored(self):
        models = {"1d": _ConstantPredictor(), "1h": _ConstantPredictor()}
        stacker = StackedStockPredictor(models, weights={"1d": 0.6, "1h": 0.4})
        assert stacker.weights["1d"] == pytest.approx(0.6)
        assert stacker.weights["1h"] == pytest.approx(0.4)

    def test_models_stored(self):
        stacker = _make_stacker()
        assert "1d" in stacker.models
        assert "1h" in stacker.models


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


class TestStackedStockPredictorTrain:
    def test_train_runs_without_error(self):
        stacker = _make_stacker()
        stacker.train(_make_train_data())

    def test_weights_updated_after_training(self):
        stacker = _make_stacker()
        stacker.train(_make_train_data())
        # Weights should have been replaced by the optimizer
        assert set(stacker.weights.keys()) == {"1d", "1h"}

    def test_weights_sum_to_one(self):
        stacker = _make_stacker()
        stacker.train(_make_train_data())
        total = sum(stacker.weights.values())
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_feature_importance_computed_after_training(self):
        stacker = _make_stacker()
        stacker.train(_make_train_data())
        assert stacker.feature_importance is not None
        assert "feature" in stacker.feature_importance.columns
        assert "importance" in stacker.feature_importance.columns


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------


class TestStackedStockPredictorPredict:
    def _trained_stacker(self):
        stacker = _make_stacker()
        stacker.train(_make_train_data())
        return stacker

    def test_predict_returns_numpy_array(self):
        stacker = self._trained_stacker()
        x_dict = {"1d": _make_feature_df(20, seed=5), "1h": _make_feature_df(20, seed=6)}
        result = stacker.predict(x_dict)
        assert isinstance(result, np.ndarray)

    def test_predict_length_equals_input_length(self):
        stacker = self._trained_stacker()
        n = 15
        x_dict = {"1d": _make_feature_df(n), "1h": _make_feature_df(n)}
        result = stacker.predict(x_dict)
        assert len(result) == n

    def test_predict_length_when_inputs_differ(self):
        """predict returns min-length array when inputs have different row counts."""
        stacker = self._trained_stacker()
        x_dict = {"1d": _make_feature_df(20), "1h": _make_feature_df(10)}
        result = stacker.predict(x_dict)
        assert len(result) == 10

    def test_predict_output_is_finite(self):
        stacker = self._trained_stacker()
        x_dict = {"1d": _make_feature_df(20), "1h": _make_feature_df(20)}
        result = stacker.predict(x_dict)
        assert np.all(np.isfinite(result))


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


class TestStackedStockPredictorEvaluate:
    def _trained_stacker_and_data(self):
        stacker = _make_stacker()
        stacker.train(_make_train_data())
        n = 20
        x_dict = {"1d": _make_feature_df(n, seed=7), "1h": _make_feature_df(n, seed=8)}
        y_true = _make_target(n, seed=7)
        return stacker, x_dict, y_true

    def test_evaluate_returns_dict_with_keys(self):
        stacker, x_dict, y_true = self._trained_stacker_and_data()
        result = stacker.evaluate(x_dict, y_true)
        for key in ("predictions", "mse", "rmse", "mae", "r2"):
            assert key in result

    def test_rmse_equals_sqrt_mse(self):
        stacker, x_dict, y_true = self._trained_stacker_and_data()
        result = stacker.evaluate(x_dict, y_true)
        assert result["rmse"] == pytest.approx(np.sqrt(result["mse"]), rel=1e-6)

    def test_mse_is_non_negative(self):
        stacker, x_dict, y_true = self._trained_stacker_and_data()
        result = stacker.evaluate(x_dict, y_true)
        assert result["mse"] >= 0


# ---------------------------------------------------------------------------
# get_model_predictions / model_prediction_correlation
# ---------------------------------------------------------------------------


class TestGetModelPredictions:
    def _trained_stacker(self):
        stacker = _make_stacker()
        stacker.train(_make_train_data())
        return stacker

    def test_returns_dict_with_model_keys(self):
        stacker = self._trained_stacker()
        x_dict = {"1d": _make_feature_df(20), "1h": _make_feature_df(20)}
        result = stacker.get_model_predictions(x_dict)
        assert set(result.keys()) == {"1d", "1h"}

    def test_all_values_are_arrays(self):
        stacker = self._trained_stacker()
        x_dict = {"1d": _make_feature_df(20), "1h": _make_feature_df(20)}
        result = stacker.get_model_predictions(x_dict)
        for arr in result.values():
            assert isinstance(arr, np.ndarray)

    def test_correlation_is_square_dataframe(self):
        stacker = self._trained_stacker()
        x_dict = {"1d": _make_feature_df(20), "1h": _make_feature_df(20)}
        corr = stacker.model_prediction_correlation(x_dict)
        assert isinstance(corr, pd.DataFrame)
        assert corr.shape == (2, 2)

    def test_diagonal_correlation_is_one_or_nan(self):
        """Diagonal should be 1.0 for varying predictions or NaN for constant predictors."""
        stacker = self._trained_stacker()
        x_dict = {"1d": _make_feature_df(20), "1h": _make_feature_df(20)}
        corr = stacker.model_prediction_correlation(x_dict)
        diag = np.diag(corr.values)
        # Each diagonal element is either exactly 1.0 (varying preds) or NaN (constant preds)
        for val in diag:
            assert np.isnan(val) or abs(val - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# save_model / load_model
# ---------------------------------------------------------------------------


class TestStackedStockPredictorSaveLoad:
    def _trained_stacker(self):
        stacker = _make_stacker()
        stacker.train(_make_train_data())
        return stacker

    def test_save_creates_file(self):
        stacker = self._trained_stacker()
        with tempfile.TemporaryDirectory() as tmpdir:
            stacker.save_model("TEST", save_path=tmpdir)
            assert os.path.isfile(os.path.join(tmpdir, "TEST_stacked_model.pkl"))

    def test_load_returns_stacked_predictor(self):
        stacker = self._trained_stacker()
        with tempfile.TemporaryDirectory() as tmpdir:
            stacker.save_model("TEST", save_path=tmpdir)
            loaded = StackedStockPredictor.load_model("TEST", save_path=tmpdir)
        assert isinstance(loaded, StackedStockPredictor)

    def test_load_preserves_model_keys(self):
        stacker = self._trained_stacker()
        with tempfile.TemporaryDirectory() as tmpdir:
            stacker.save_model("TEST", save_path=tmpdir)
            loaded = StackedStockPredictor.load_model("TEST", save_path=tmpdir)
        assert set(loaded.models.keys()) == {"1d", "1h"}


# ---------------------------------------------------------------------------
# Rolling weight recalibration
# ---------------------------------------------------------------------------


class TestRollingRecalibration:
    def _stacker_with_buffers(self, window=10, freq=5):
        """Return a trained stacker with small rolling_window and recalibration_freq."""
        models = {"1d": _ConstantPredictor(), "1h": _ConstantPredictor()}
        stacker = StackedStockPredictor(models, rolling_window=window, recalibration_freq=freq)
        stacker.train(_make_train_data())
        return stacker

    def test_update_returns_false_before_buffer_full(self):
        stacker = self._stacker_with_buffers(window=10, freq=5)
        for _ in range(9):
            result = stacker.update({"1d": 0.01, "1h": 0.02}, actual=0.015)
        assert result is False

    def test_update_returns_true_when_recalibration_triggers(self):
        stacker = self._stacker_with_buffers(window=5, freq=5)
        triggered = False
        for _ in range(10):
            result = stacker.update({"1d": 0.01, "1h": 0.02}, actual=0.015)
            if result:
                triggered = True
        assert triggered

    def test_buffer_capped_at_rolling_window(self):
        stacker = self._stacker_with_buffers(window=5, freq=100)
        for i in range(20):
            stacker.update({"1d": float(i), "1h": float(i)}, actual=float(i))
        assert len(stacker._actual_buffer) == 5
        assert len(stacker._pred_buffer["1d"]) == 5

    def test_weights_update_after_recalibration(self):
        stacker = self._stacker_with_buffers(window=5, freq=5)
        for _ in range(5):
            stacker.update({"1d": 0.01, "1h": -0.05}, actual=0.01)
        # Weights may or may not change (depends on optimiser), but should be valid
        assert set(stacker.weights.keys()) == {"1d", "1h"}
        total = sum(stacker.weights.values())
        assert abs(total - 1.0) < 1e-6

    def test_recalibrated_weights_sum_to_one(self):
        stacker = self._stacker_with_buffers(window=5, freq=5)
        for _ in range(10):
            stacker.update({"1d": 0.01, "1h": 0.02}, actual=0.015)
        assert abs(sum(stacker.weights.values()) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Per-regime weights
# ---------------------------------------------------------------------------


class TestPerRegimeWeights:
    def _trained_stacker(self):
        models = {"1d": _ConstantPredictor(), "1h": _ConstantPredictor()}
        stacker = StackedStockPredictor(models)
        stacker.train(_make_train_data())
        return stacker

    def test_set_and_get_weights_for_regime(self):
        stacker = self._trained_stacker()
        stacker.set_weights_for_regime(0, {"1d": 0.7, "1h": 0.3})
        w = stacker.get_weights_for_regime(0)
        assert w["1d"] == pytest.approx(0.7)
        assert w["1h"] == pytest.approx(0.3)

    def test_get_weights_for_unknown_regime_falls_back_to_global(self):
        stacker = self._trained_stacker()
        w = stacker.get_weights_for_regime(99)
        assert w is stacker.weights

    def test_calibrate_regime_weights_stores_per_regime(self):
        n = 60
        stacker = self._trained_stacker()
        x_val_dict = {
            "1d": _make_feature_df(n, seed=10),
            "1h": _make_feature_df(n, seed=11),
        }
        y_val = _make_target(n, seed=10)
        # Alternate regimes: 0 for first half, 1 for second half
        regime_labels = pd.Series([0] * 30 + [1] * 30)
        stacker.calibrate_regime_weights(x_val_dict, y_val, regime_labels)
        assert 0 in stacker._weights_per_regime
        assert 1 in stacker._weights_per_regime

    def test_calibrated_regime_weights_sum_to_one(self):
        n = 60
        stacker = self._trained_stacker()
        x_val_dict = {
            "1d": _make_feature_df(n, seed=10),
            "1h": _make_feature_df(n, seed=11),
        }
        y_val = _make_target(n, seed=10)
        regime_labels = pd.Series([0] * 30 + [1] * 30)
        stacker.calibrate_regime_weights(x_val_dict, y_val, regime_labels)
        for regime, w in stacker._weights_per_regime.items():
            assert abs(sum(w.values()) - 1.0) < 1e-6, f"Regime {regime} weights don't sum to 1"

    def test_calibrate_skips_regime_with_too_few_samples(self):
        stacker = self._trained_stacker()
        x_val_dict = {"1d": _make_feature_df(20), "1h": _make_feature_df(20)}
        y_val = _make_target(20)
        # Regime 99 has only 1 sample — below min required
        regime_labels = pd.Series([0] * 19 + [99])
        stacker.calibrate_regime_weights(x_val_dict, y_val, regime_labels)
        assert 99 not in stacker._weights_per_regime

    def test_predict_with_regime_returns_numpy_array(self):
        stacker = self._trained_stacker()
        stacker.set_weights_for_regime(0, {"1d": 0.8, "1h": 0.2})
        n = 10
        x_dict = {"1d": _make_feature_df(n), "1h": _make_feature_df(n)}
        result = stacker.predict_with_regime(x_dict, regime=0)
        assert isinstance(result, np.ndarray)
        assert len(result) == n

    def test_predict_with_regime_falls_back_for_unknown_regime(self):
        stacker = self._trained_stacker()
        n = 10
        x_dict = {"1d": _make_feature_df(n), "1h": _make_feature_df(n)}
        # No regime 42 configured — should fall back to global weights == same as predict()
        result_regime = stacker.predict_with_regime(x_dict, regime=42)
        result_default = stacker.predict(x_dict)
        np.testing.assert_array_equal(result_regime, result_default)

    def test_predict_with_regime_does_not_mutate_global_weights(self):
        stacker = self._trained_stacker()
        stacker.set_weights_for_regime(0, {"1d": 0.9, "1h": 0.1})
        original_weights = dict(stacker.weights)
        n = 10
        x_dict = {"1d": _make_feature_df(n), "1h": _make_feature_df(n)}
        stacker.predict_with_regime(x_dict, regime=0)
        assert stacker.weights == original_weights
