import os
import pickle
from typing import Dict, Any
import numpy as np
import pandas as pd

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.optimize import minimize

from src.util import logger


class StackedStockPredictor:
    """
    Stacked predictor combining multiple base models (e.g., daily + hourly).
    """

    def __init__(self, models: Dict[str, Any], weights: Dict[str, float] = None):
        """
        :param models: dict of models, e.g., {"daily": XGBoostStockPredictor(), "hourly": LightGBMStockPredictor()}
        :param weights: optional dict of weights for stacking
        """
        self.models = models
        self.feature_importance = None
        self.weights = weights if weights is not None else {k: 1.0 for k in models}

    def _optimize_weights(self, x_val_dict, y_val):
        """
        Find optimal linear combination of model predictions on a validation set
        """
        valid_predictions = {}

        # Ensure y_val is a Series with proper index
        if not isinstance(y_val, pd.Series):
            y_val = pd.Series(y_val)

        reference_index = y_val.index
        target_length = len(y_val)

        for name, model in self.models.items():
            pred = model.predict(x_val_dict[name])
            pred = np.asarray(pred).ravel().astype(float)  # Ensure numeric dtype

            # Align predictions to reference timeline length
            if len(pred) != target_length:
                if len(pred) > target_length:
                    # Downsample: take every nth element
                    step = len(pred) // target_length
                    pred_aligned = pred[::step][:target_length]
                else:
                    # Upsample: forward fill with numpy
                    pred_aligned = np.zeros(target_length)
                    pred_aligned[: len(pred)] = pred
                    pred_aligned[len(pred) :] = pred[-1]  # Forward fill last value
            else:
                pred_aligned = pred

            valid_predictions[name] = pred_aligned

        def objective(weights):
            weighted_pred = sum(w * valid_predictions[name] for w, name in zip(weights, self.models.keys()))
            return mean_squared_error(y_val, weighted_pred)

        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in self.models]
        initial = np.array([1 / len(self.models)] * len(self.models))

        result = minimize(objective, initial, bounds=bounds, constraints=constraints)

        return {name: w for name, w in zip(self.models.keys(), result.x)}

    def train(self, train_data: Dict[str, tuple]):
        """
        Train each base model.
        :param train_data: dict of (x_train, y_train, x_val, y_val) tuples per model
        """
        model_r2_scores = {}

        x_val_dict = {}
        y_val_combined = None

        for name, model in self.models.items():
            x_train, y_train, x_val, y_val = train_data[name]
            model.train(x_train, y_train, x_val, y_val)

            y_pred = model.predict(x_val)
            r2 = r2_score(y_val, y_pred)
            model_r2_scores[name] = r2
            logger.info(f"{name} model trained | R\u00b2 (val): {r2:.4f}")

            x_val_dict[name] = x_val
            if y_val_combined is None:
                y_val_combined = y_val

        self.weights = self._optimize_weights(x_val_dict, y_val_combined)

        logger.info("Final model weights (optimized on validation set):")
        for k, w in self.weights.items():
            logger.info(f"  {k}: {w:.3f}")

        self.feature_importance = self._compute_feature_importance()

    def predict(self, x_dict: dict) -> np.ndarray:
        """
        Return 1D stacked predictions
        """
        preds = []

        # Collect raw predictions once, then align to the minimum length
        raw_predictions = {}
        for name, model in self.models.items():
            raw_predictions[name] = np.asarray(model.predict(x_dict[name])).ravel()

        if not raw_predictions:
            return np.array([])

        min_length = min(len(p) for p in raw_predictions.values())

        for name, p in raw_predictions.items():
            # Align to minimum length
            if len(p) >= min_length:
                p_aligned = p[:min_length]
            else:
                p_aligned = np.pad(p, (0, min_length - len(p)), mode="constant", constant_values=0)
            preds.append(p_aligned * self.weights.get(name, 1.0))

        # Now all predictions have the same length
        stacked_preds = np.sum(preds, axis=0) / sum(self.weights.values())

        return np.asarray(stacked_preds).ravel()

    def get_model_predictions(self, x_dict: Dict[str, Any]) -> Dict[str, np.ndarray]:
        predictions = {}
        for name, model in self.models.items():
            p = model.predict(x_dict[name])
            predictions[name] = np.asarray(p).ravel()
        return predictions

    def model_prediction_correlation(self, x_dict: Dict[str, Any]) -> pd.DataFrame:
        predictions = self.get_model_predictions(x_dict)

        # Find minimum length to align all predictions
        min_length = min(len(pred) for pred in predictions.values())

        # Truncate all predictions to same length
        aligned_predictions = {name: pred[:min_length] for name, pred in predictions.items()}

        df = pd.DataFrame(aligned_predictions)
        return df.corr()

    def evaluate(self, x_dict: Dict[str, Any], y_true) -> dict:
        """
        Evaluate stacked model.
        """
        predictions = self.predict(x_dict)

        prediction_correlation = self.model_prediction_correlation(x_dict)
        mse = mean_squared_error(y_true, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, predictions)
        r2 = r2_score(y_true, predictions)

        logger.info("Stacked Model Performance:")
        logger.info(f"  MSE : {mse:.8f}")
        logger.info(f"  RMSE: {rmse:.8f}")
        logger.info(f"  MAE : {mae:.8f}")
        logger.info(f"  R\u00b2  : {r2:.8f}")
        logger.debug(f"Prediction Correlation:\n {prediction_correlation}")

        return {
            "predictions": predictions,
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
        }

    def save_model(self, symbol, save_path="models"):
        os.makedirs(save_path, exist_ok=True)
        filename = f"{save_path}/{symbol}_stacked_model.pkl"

        feature_columns_per_model = {}
        for name, model in self.models.items():
            if hasattr(model, "feature_columns") and model.feature_columns is not None:
                feature_columns_per_model[name] = model.feature_columns

        model_data = {"stacked_predictor": self, "feature_columns_per_model": feature_columns_per_model}

        with open(filename, "wb") as f:
            pickle.dump(model_data, f)

        logger.info(f"Saved full stacked model to {filename}")
        for name, cols in feature_columns_per_model.items():
            logger.info(f"Saved {len(cols)} feature columns for '{name}' model")

    @staticmethod
    def load_model(symbol, save_path="models"):
        filename = f"{save_path}/{symbol}_stacked_model.pkl"
        with open(filename, "rb") as f:
            model_data = pickle.load(f)

        if isinstance(model_data, dict) and "stacked_predictor" in model_data:
            predictor = model_data["stacked_predictor"]
            feature_columns_per_model = model_data.get("feature_columns_per_model", {})
            logger.info(f"Loaded stacked model from {filename}")
            for name, cols in feature_columns_per_model.items():
                logger.debug(f"Model '{name}' expects {len(cols)} features in specific order")
        else:
            predictor = model_data
            logger.info(f"Loaded stacked model from {filename} (old format)")

        return predictor

    def _compute_feature_importance(self):
        combined = None
        for name, model in self.models.items():
            if hasattr(model, "feature_importance") and model.feature_importance is not None:
                fi = model.feature_importance.copy()
                weight = self.weights.get(name, 1.0)
                fi["importance"] *= weight
                fi.rename(columns={"importance": f"importance_{name}"}, inplace=True)
                if combined is None:
                    combined = fi
                else:
                    combined = combined.merge(fi, on="feature", how="outer")
        if combined is not None:
            combined = combined.fillna(0)
            importance_cols = [c for c in combined.columns if c.startswith("importance_")]
            combined["importance"] = combined[importance_cols].sum(axis=1)
            return combined[["feature", "importance"]].sort_values("importance", ascending=False)
        return None
