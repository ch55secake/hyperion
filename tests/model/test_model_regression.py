"""Regression tests for XGBoost, LightGBM, and StackedStockPredictor models."""

import numpy as np
import pytest
from sklearn.metrics import r2_score

from src.model import XGBoostStockPredictor, LightGBMStockPredictor, StackedStockPredictor
from tests.helpers import fast_xgb_params, fast_lgb_params


class TestXGBoostRegression:
    """Regression tests for XGBoostStockPredictor."""

    def test_xgb_predictions_deterministic(self, trained_xgb_model):
        """XGBoost with fixed seed produces identical predictions across two training runs."""
        model1, x_train, x_test, y_train, _ = trained_xgb_model

        model2 = XGBoostStockPredictor(params=fast_xgb_params())
        model2.train(x_train, y_train)

        preds1 = model1.predict(x_test)
        preds2 = model2.predict(x_test)

        np.testing.assert_array_almost_equal(
            preds1,
            preds2,
            decimal=5,
            err_msg="XGBoost predictions differ between two runs with the same seed",
        )

    def test_xgb_r2_above_minimum_threshold(self, trained_xgb_model):
        """Model should achieve R² > -2.0 on synthetic data (model is not completely degenerate)."""
        model, _, x_test, _, y_test = trained_xgb_model
        preds = model.predict(x_test)
        r2 = r2_score(y_test, preds)
        assert r2 > -2.0, f"XGBoost R² is unexpectedly low: {r2:.4f}"

    def test_xgb_predictions_are_finite(self, trained_xgb_model):
        """XGBoost predictions must not contain NaN or Inf."""
        model, _, x_test, _, _ = trained_xgb_model
        preds = model.predict(x_test)
        assert np.all(np.isfinite(preds)), "XGBoost predictions contain NaN or Inf values"

    def test_xgb_prediction_length_matches_input(self, trained_xgb_model):
        """Number of predictions must equal the number of test rows."""
        model, _, x_test, _, _ = trained_xgb_model
        preds = model.predict(x_test)
        assert len(preds) == len(x_test), f"Prediction count {len(preds)} != test row count {len(x_test)}"

    def test_xgb_evaluate_returns_expected_keys(self, trained_xgb_model):
        """evaluate() must return a dict with standard metric keys."""
        model, _, x_test, _, y_test = trained_xgb_model
        result = model.evaluate(x_test, y_test)
        for key in ("r2", "rmse", "mae", "mse", "predictions"):
            assert key in result, f"Key '{key}' missing from evaluate() result"


class TestLightGBMRegression:
    """Regression tests for LightGBMStockPredictor."""

    def test_lgb_predictions_deterministic(self, trained_lgb_model):
        """LightGBM with fixed seed produces identical predictions across two training runs."""
        model1, x_train, x_test, y_train, _ = trained_lgb_model

        model2 = LightGBMStockPredictor(params=fast_lgb_params())
        model2.train(x_train, y_train)

        preds1 = model1.predict(x_test)
        preds2 = model2.predict(x_test)

        np.testing.assert_array_almost_equal(
            preds1,
            preds2,
            decimal=5,
            err_msg="LightGBM predictions differ between two runs with the same seed",
        )

    def test_lgb_r2_above_minimum_threshold(self, trained_lgb_model):
        """LightGBM model should achieve R² > -2.0 on synthetic data (model is not completely degenerate)."""
        model, _, x_test, _, y_test = trained_lgb_model
        preds = model.predict(x_test)
        r2 = r2_score(y_test, preds)
        assert r2 > -2.0, f"LightGBM R² is unexpectedly low: {r2:.4f}"

    def test_lgb_predictions_are_finite(self, trained_lgb_model):
        """LightGBM predictions must not contain NaN or Inf."""
        model, _, x_test, _, _ = trained_lgb_model
        preds = model.predict(x_test)
        assert np.all(np.isfinite(preds)), "LightGBM predictions contain NaN or Inf values"

    def test_lgb_prediction_length_matches_input(self, trained_lgb_model):
        """Number of predictions must equal the number of test rows."""
        model, _, x_test, _, _ = trained_lgb_model
        preds = model.predict(x_test)
        assert len(preds) == len(x_test), f"Prediction count {len(preds)} != test row count {len(x_test)}"

    def test_lgb_evaluate_returns_expected_keys(self, trained_lgb_model):
        """evaluate() must return a dict with standard metric keys."""
        model, _, x_test, _, y_test = trained_lgb_model
        result = model.evaluate(x_test, y_test)
        for key in ("r2", "rmse", "mae", "mse", "predictions"):
            assert key in result, f"Key '{key}' missing from evaluate() result"


class TestWeightedEnsembleRegression:
    """Regression tests for StackedStockPredictor (weighted ensemble)."""

    def test_ensemble_weights_sum_to_one(self, trained_xgb_model, trained_lgb_model):
        """Optimized weights must sum to 1.0 within tolerance."""
        xgb_model, x_train, x_test, y_train, y_test = trained_xgb_model

        stacked = StackedStockPredictor(
            {
                "xgb": XGBoostStockPredictor(params=fast_xgb_params()),
                "lgb": LightGBMStockPredictor(params=fast_lgb_params()),
            }
        )

        train_data = {
            "xgb": (x_train, y_train, x_test, y_test),
            "lgb": (x_train, y_train, x_test, y_test),
        }
        stacked.train(train_data)

        total_weight = sum(stacked.weights.values())
        assert abs(total_weight - 1.0) < 1e-6, f"Weights sum to {total_weight}, expected 1.0"

    def test_ensemble_improves_over_worst_model(self, trained_xgb_model, trained_lgb_model):
        """Ensemble R² should be >= worst individual model R² (within tolerance)."""
        xgb_model, x_train, x_test, y_train, y_test = trained_xgb_model
        lgb_model, _, _, _, _ = trained_lgb_model

        xgb_r2 = r2_score(y_test, xgb_model.predict(x_test))
        lgb_r2 = r2_score(y_test, lgb_model.predict(x_test))
        worst_r2 = min(xgb_r2, lgb_r2)

        stacked = StackedStockPredictor(
            {
                "xgb": XGBoostStockPredictor(params=fast_xgb_params()),
                "lgb": LightGBMStockPredictor(params=fast_lgb_params()),
            }
        )
        train_data = {
            "xgb": (x_train, y_train, x_test, y_test),
            "lgb": (x_train, y_train, x_test, y_test),
        }
        stacked.train(train_data)

        ensemble_preds = stacked.predict({"xgb": x_test, "lgb": x_test})
        ensemble_r2 = r2_score(y_test, ensemble_preds)

        assert (
            ensemble_r2 >= worst_r2 - 0.05
        ), f"Ensemble R² ({ensemble_r2:.4f}) is worse than worst model R² ({worst_r2:.4f}) by more than tolerance"

    def test_ensemble_predictions_are_finite(self, trained_xgb_model, trained_lgb_model):
        """StackedStockPredictor predictions must not contain NaN or Inf."""
        xgb_model, x_train, x_test, y_train, y_test = trained_xgb_model

        stacked = StackedStockPredictor(
            {
                "xgb": XGBoostStockPredictor(params=fast_xgb_params()),
                "lgb": LightGBMStockPredictor(params=fast_lgb_params()),
            }
        )
        train_data = {
            "xgb": (x_train, y_train, x_test, y_test),
            "lgb": (x_train, y_train, x_test, y_test),
        }
        stacked.train(train_data)

        preds = stacked.predict({"xgb": x_test, "lgb": x_test})
        assert np.all(np.isfinite(preds)), "Ensemble predictions contain NaN or Inf values"
