"""Unit tests for src/model/model.py (base Model class)."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.model.model import Model

# ---------------------------------------------------------------------------
# Minimal concrete implementation of the abstract Model
# ---------------------------------------------------------------------------


class _DummyModel(Model):
    """Trivial concrete subclass: simply stores and returns a constant."""

    def __init__(self, model_name: str = "dummy", params=None):
        super().__init__(model_name, params=params)
        self._constant = None
        self.model = None

    def train(self, x_train, y_train, x_val=None, y_val=None):
        self._prepare_columns(x_train)
        self.model = True  # mark as trained
        self._constant = float(np.mean(y_train))
        return x_val, y_val  # suppress unused-argument warning

    def predict(self, x):
        x_processed = self._prepare_prediction(x)
        return np.full(len(x_processed), self._constant)


def _make_numeric_df(n=50, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "feature_a": rng.normal(0, 1, n),
            "feature_b": rng.uniform(0, 1, n),
        }
    )


def _make_target(n=50, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 0.1, n)


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------


class TestModelInit:
    def test_default_params(self):
        m = _DummyModel()
        assert m.model_name == "dummy"
        assert m.model is None
        assert m.feature_columns is None

    def test_custom_params(self):
        m = _DummyModel(params={"lr": 0.01})
        assert m.params == {"lr": 0.01}


# ---------------------------------------------------------------------------
# train / predict
# ---------------------------------------------------------------------------


class TestTrainPredict:
    def test_train_sets_model(self):
        m = _DummyModel()
        x = _make_numeric_df()
        y = _make_target()
        m.train(x, y)
        assert m.model is not None

    def test_predict_after_train(self):
        m = _DummyModel()
        x = _make_numeric_df()
        y = _make_target()
        m.train(x, y)
        preds = m.predict(x)
        assert len(preds) == len(x)

    def test_predict_before_train_raises(self):
        m = _DummyModel()
        x = _make_numeric_df()
        with pytest.raises(ValueError, match="Model not trained yet"):
            m.predict(x)

    def test_predict_is_constant(self):
        m = _DummyModel()
        x = _make_numeric_df()
        y = np.ones(50) * 3.14
        m.train(x, y)
        preds = m.predict(x)
        assert np.allclose(preds, 3.14)


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


class TestEvaluate:
    def test_returns_dict_with_keys(self):
        m = _DummyModel()
        x = _make_numeric_df()
        y = _make_target()
        m.train(x, y)
        result = m.evaluate(x, y)
        assert "mse" in result
        assert "rmse" in result
        assert "mae" in result
        assert "r2" in result
        assert "predictions" in result

    def test_perfect_prediction_has_zero_mse(self):
        m = _DummyModel()
        x = _make_numeric_df()
        y = np.ones(50) * 5.0
        m.train(x, y)
        result = m.evaluate(x, y)
        assert abs(result["mse"]) < 1e-10

    def test_rmse_equals_sqrt_mse(self):
        m = _DummyModel()
        x = _make_numeric_df()
        y = _make_target()
        m.train(x, y)
        result = m.evaluate(x, y)
        assert abs(result["rmse"] - np.sqrt(result["mse"])) < 1e-10


# ---------------------------------------------------------------------------
# save_model / load_model
# ---------------------------------------------------------------------------


@pytest.mark.regression
class TestSaveLoadModel:
    def test_save_creates_file(self):
        m = _DummyModel()
        x = _make_numeric_df()
        y = _make_target()
        m.train(x, y)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = m.save_model("TEST", save_path=tmpdir)
            assert os.path.isfile(path)

    def test_save_raises_without_training(self):
        m = _DummyModel()
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="No model to save"):
                m.save_model("TEST", save_path=tmpdir)

    def test_load_model_file_not_found(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                _DummyModel.load_model("dummy", "MISSING", load_path=tmpdir)

    def test_save_and_load_roundtrip(self):
        m = _DummyModel()
        x = _make_numeric_df()
        y = np.ones(50) * 7.0
        m.train(x, y)
        with tempfile.TemporaryDirectory() as tmpdir:
            m.save_model("AAPL", save_path=tmpdir)
            loaded = _DummyModel.load_model("dummy", "AAPL", load_path=tmpdir)
        assert loaded.feature_columns == m.feature_columns
        assert loaded.numeric_columns == m.numeric_columns


# ---------------------------------------------------------------------------
# _prepare_columns helpers
# ---------------------------------------------------------------------------


class TestPrepareColumns:
    def test_stores_feature_columns(self):
        m = _DummyModel()
        x = _make_numeric_df()
        y = _make_target()
        m.train(x, y)
        assert m.feature_columns == ["feature_a", "feature_b"]

    def test_numeric_columns_detected(self):
        m = _DummyModel()
        x = _make_numeric_df()
        y = _make_target()
        m.train(x, y)
        assert "feature_a" in m.numeric_columns
        assert "feature_b" in m.numeric_columns

    def test_object_columns_converted_to_category(self):
        m = _DummyModel()
        x = pd.DataFrame(
            {
                "num": [1.0, 2.0, 3.0],
                "cat": ["a", "b", "c"],
            }
        )
        y = np.array([1.0, 2.0, 3.0])
        m.train(x, y)
        # Object column should have been promoted to categorical
        assert "cat" in m.categorical_columns
