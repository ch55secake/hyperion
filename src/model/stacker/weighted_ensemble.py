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

    Supports three adaptive weighting mechanisms:

    1. **Rolling weight recalibration** — call ``update(predictions_dict, actual)`` after every
       live prediction to accumulate a rolling buffer.  Once the buffer reaches *rolling_window*
       observations, weights are re-optimised automatically every *recalibration_freq* steps.

    2. **Per-regime weights** — call ``calibrate_regime_weights(x_val_dict, y_val, regime_labels)``
       to learn separate blend weights for each detected market regime, then use
       ``predict_with_regime(x_dict, regime)`` at inference time.

    3. **Standard training-time optimisation** — the original ``train()`` call still optimises
       weights on the validation set and stores them in ``self.weights``.
    """

    def __init__(
        self,
        models: Dict[str, Any],
        weights: Dict[str, float] | None = None,
        rolling_window: int = 50,
        recalibration_freq: int = 10,
    ):
        """
        :param models: dict of models, e.g., {"daily": XGBoostStockPredictor(), "hourly": LightGBMStockPredictor()}
        :param weights: optional dict of initial weights for blending
        :param rolling_window: number of recent observations kept for rolling recalibration
        :param recalibration_freq: number of new observations between successive recalibrations
        """
        self.models = models
        self.feature_importance = None
        self.weights = weights if weights is not None else {k: 1.0 for k in models}
        self.rolling_window = rolling_window
        self.recalibration_freq = recalibration_freq

        # Rolling recalibration state
        self._pred_buffer: Dict[str, list] = {k: [] for k in models}
        self._actual_buffer: list = []
        self._steps_since_recalibration: int = 0

        # Per-regime weights: {regime_label -> {model_name -> weight}}
        self._weights_per_regime: Dict[int, Dict[str, float]] = {}

    def _optimize_weights(self, x_val_dict, y_val):
        """
        Find optimal linear combination of model predictions on a validation set
        """
        valid_predictions = {}

        # Ensure y_val is a Series with proper index
        if not isinstance(y_val, pd.Series):
            y_val = pd.Series(y_val)

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

    # ------------------------------------------------------------------
    # Rolling weight recalibration
    # ------------------------------------------------------------------

    def update(self, predictions_dict: Dict[str, float], actual: float) -> bool:
        """
        Record a new live observation and recalibrate blend weights if due.

        Call this method once per prediction step during simulation or live inference,
        passing the scalar prediction from each base model and the observed actual value.
        Weights are re-optimised automatically once the rolling buffer is full and
        *recalibration_freq* new observations have been collected since the last update.

        :param predictions_dict: dict mapping model name to its scalar prediction for this step
        :param actual: the observed actual value (e.g. realised return) for this step
        :return: ``True`` if weights were recalibrated this step, ``False`` otherwise
        """
        for name in self.models:
            self._pred_buffer[name].append(float(predictions_dict[name]))
            if len(self._pred_buffer[name]) > self.rolling_window:
                self._pred_buffer[name] = self._pred_buffer[name][-self.rolling_window :]

        self._actual_buffer.append(float(actual))
        if len(self._actual_buffer) > self.rolling_window:
            self._actual_buffer = self._actual_buffer[-self.rolling_window :]

        self._steps_since_recalibration += 1

        if len(self._actual_buffer) >= self.rolling_window and self._steps_since_recalibration >= self.recalibration_freq:
            self._recalibrate_from_buffer()
            self._steps_since_recalibration = 0
            return True
        return False

    def _recalibrate_from_buffer(self) -> None:
        """Re-optimise blend weights using the current rolling prediction/actual buffer."""
        pred_arrays = {name: np.array(self._pred_buffer[name]) for name in self.models}
        actuals = np.array(self._actual_buffer)

        def objective(weights):
            weighted = sum(w * pred_arrays[name] for w, name in zip(weights, self.models.keys()))
            return mean_squared_error(actuals, weighted)

        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in self.models]
        raw = np.array(list(self.weights.values()))
        initial = raw / raw.sum() if raw.sum() > 0 else np.full(len(self.models), 1.0 / len(self.models))

        result = minimize(objective, initial, bounds=bounds, constraints=constraints)
        new_weights = {name: float(w) for name, w in zip(self.models.keys(), result.x)}

        logger.info("Rolling recalibration: updated blend weights from buffer of %d observations:", len(actuals))
        for k, w in new_weights.items():
            logger.info("  %s: %.3f", k, w)

        self.weights = new_weights

    # ------------------------------------------------------------------
    # Per-regime weights
    # ------------------------------------------------------------------

    def get_weights_for_regime(self, regime: int) -> Dict[str, float]:
        """
        Return blend weights for *regime*, falling back to the current global weights if
        no regime-specific weights have been calibrated for that regime label.
        """
        return self._weights_per_regime.get(regime, self.weights)

    def set_weights_for_regime(self, regime: int, weights: Dict[str, float]) -> None:
        """
        Manually assign blend weights for a specific market regime label.

        :param regime: integer regime label (must be consistent with your RegimeDetector)
        :param weights: dict mapping model name to weight (values need not sum to 1;
                        they will be normalised at prediction time via the standard path)
        """
        self._weights_per_regime[regime] = weights
        logger.info("Regime %d weights updated: %s", regime, {k: f"{v:.3f}" for k, v in weights.items()})

    def calibrate_regime_weights(
        self,
        x_val_dict: Dict[str, Any],
        y_val: Any,
        regime_labels: "pd.Series",
    ) -> None:
        """
        Optimise blend weights separately for each regime found in *regime_labels*.

        :param x_val_dict: dict of {model_name: validation feature DataFrame}
        :param y_val: validation target (Series or array, aligned to x_val rows)
        :param regime_labels: pd.Series of integer regime labels indexed like y_val
        """
        if not isinstance(y_val, pd.Series):
            y_val = pd.Series(y_val)

        unique_regimes = regime_labels.dropna().unique()
        for regime in unique_regimes:
            regime = int(regime)
            mask = (regime_labels == regime).values
            indices = np.where(mask)[0]
            if len(indices) < max(len(self.models), 2):
                logger.warning(
                    "Regime %d has only %d samples; skipping weight optimisation",
                    regime,
                    len(indices),
                )
                continue

            y_regime = y_val.iloc[indices]
            x_regime = {
                name: x.iloc[indices] if hasattr(x, "iloc") else x[indices] for name, x in x_val_dict.items()
            }
            weights = self._optimize_weights(x_regime, y_regime)
            self._weights_per_regime[regime] = weights
            logger.info(
                "Regime %d calibrated weights: %s",
                regime,
                {k: f"{v:.3f}" for k, v in weights.items()},
            )

    def predict_with_regime(self, x_dict: dict, regime: int) -> np.ndarray:
        """
        Return stacked predictions using the blend weights for *regime*.

        If no regime-specific weights have been calibrated for *regime*, falls back to
        the current global ``self.weights``.

        :param x_dict: dict of {model_name: feature DataFrame} for inference
        :param regime: integer regime label from a RegimeDetector
        :return: 1-D numpy array of ensemble predictions
        """
        regime_weights = self.get_weights_for_regime(regime)
        original_weights = self.weights
        self.weights = regime_weights
        try:
            result = self.predict(x_dict)
        finally:
            self.weights = original_weights
        return result

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
