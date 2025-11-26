import os
import pickle
from typing import Dict, Any
import numpy as np

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.optimize import minimize

from ..model import Model
from typing_extensions import override


class StackedStockPredictor(Model):
    """
    Stacked predictor combining multiple base models (e.g., daily + hourly).
    """

    def __init__(self, models: Dict[str, Any], weights: Dict[str, float] = None):
        """
        :param models: dict of models, e.g., {"daily": XGBoostStockPredictor(), "hourly": LightGBMStockPredictor()}
        :param weights: optional dict of weights for stacking
        """
        self.models = models
        self.weights = weights or {k: 1.0 for k in models}
        self.feature_importance = None

    def _optimize_weights(self, x_val_dict, y_val):
        """Find optimal linear combination of model predictions on validation set"""

        val_preds = {}
        for name, model in self.models.items():
            val_preds[name] = model.predict(x_val_dict[name])

        def objective(weights):
            weighted_pred = sum(w * val_preds[name] for w, name in zip(weights, self.models.keys()))
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
            print(f"✓ {name} model trained | R² (val): {r2:.4f}")

            x_val_dict[name] = x_val
            if y_val_combined is None:
                y_val_combined = y_val

        self.weights = self._optimize_weights(x_val_dict, y_val_combined)

        print("\nFinal model weights (optimized on validation set):")
        for k, w in self.weights.items():
            print(f"  {k}: {w:.3f}")

        self.feature_importance = self._compute_feature_importance()

    def predict(self, x_dict: dict) -> np.ndarray:
        """
        Return 1D stacked predictions
        """
        preds = []
        for name, model in self.models.items():
            p = model.predict(x_dict[name])
            p = np.asarray(p).ravel()
            preds.append(p * self.weights.get(name, 1.0))

        stacked_preds = np.sum(preds, axis=0) / sum(self.weights.values())

        return np.asarray(stacked_preds).ravel()

    def evaluate(self, x_dict: Dict[str, Any], y_true) -> dict:
        """
        Evaluate stacked model.
        """
        preds = self.predict(x_dict)

        mse = mean_squared_error(y_true, preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, preds)
        r2 = r2_score(y_true, preds)

        print("Stacked Model Performance:")
        print(f"  MSE : {mse:.8f}")
        print(f"  RMSE: {rmse:.8f}")
        print(f"  MAE : {mae:.8f}")
        print(f"  R²  : {r2:.8f}")

        return {
            "predictions": preds,
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
        }

    @override
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

        print(f"✓ Saved full stacked model to {filename}")
        for name, cols in feature_columns_per_model.items():
            print(f"✓ Saved {len(cols)} feature columns for '{name}' model")

    @override
    @staticmethod
    def load_model(symbol, save_path="models"):
        filename = f"{save_path}/{symbol}_stacked_model.pkl"
        with open(filename, "rb") as f:
            model_data = pickle.load(f)

        if isinstance(model_data, dict) and "stacked_predictor" in model_data:
            predictor = model_data["stacked_predictor"]
            feature_columns_per_model = model_data.get("feature_columns_per_model", {})
            print(f"✓ Loaded stacked model from {filename}")
            for name, cols in feature_columns_per_model.items():
                print(f"✓ Model '{name}' expects {len(cols)} features in specific order")
        else:
            predictor = model_data
            print(f"✓ Loaded stacked model from {filename} (old format)")

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
