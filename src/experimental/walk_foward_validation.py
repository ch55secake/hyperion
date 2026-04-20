from typing import Any, Callable, Literal

import numpy as np
import pandas as pd
from pandas import Series
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.model import XGBoostStockPredictor
from src.util import logger
from src.visualisation import generate_walk_forward_plots

WALK_FORWARD_TRAIN_WINDOW = 180  # ~9 months of training data (reduced from 252)
WALK_FORWARD_TEST_WINDOW = 20  # ~1 month of testing
WALK_FORWARD_RETRAIN_FREQ = 20  # Retrain every ~1 month

WindowType = Literal["expanding", "sliding"]

_ANNUALISATION_FACTOR = np.sqrt(252)
_DEGRADATION_R2_DROP_THRESHOLD = 0.5  # flag when R² falls >50 % relative to the first fold


class WalkForwardValidator:
    """Implements walk-forward analysis for time series.

    Supports two window strategies:

    * ``"expanding"`` (default) — the training set grows with each fold: all
      data up to the current split point is used for training.  This mirrors
      the standard expanding-window cross-validation approach.
    * ``"sliding"`` — the training window has a fixed size (``train_window``).
      Older observations are dropped as newer ones are added, which helps
      avoid down-weighting recent market regimes.
    """

    def __init__(
        self,
        train_window: int = 252,
        test_window: int = 21,
        retrain_frequency: int = 21,
        window_type: WindowType = "expanding",
    ):
        """
        Args:
            train_window: Number of rows to use for the initial training window.
                For ``"sliding"`` mode this is also the fixed window size.
            test_window: Number of rows to evaluate on before retraining.
            retrain_frequency: How many rows to advance the training end between
                consecutive folds.
            window_type: ``"expanding"`` keeps start fixed at 0; ``"sliding"``
                advances the start by ``retrain_frequency`` at each fold.
        """
        if window_type not in ("expanding", "sliding"):
            raise ValueError(f"window_type must be 'expanding' or 'sliding', got '{window_type}'")
        self.train_window = train_window
        self.test_window = test_window
        self.retrain_frequency = retrain_frequency
        self.window_type = window_type
        self.fold_results: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Split generation
    # ------------------------------------------------------------------

    def split(self, x: pd.DataFrame, dates: Any) -> list[dict[str, Any]]:
        """Generate train/test index splits for walk-forward validation.

        Returns a list of fold dicts, each containing:
            * ``fold`` — 1-based fold number
            * ``train_indices`` / ``test_indices`` — integer position lists
            * ``train_dates`` / ``test_dates`` — (start, end) date tuples
        """
        splits: list[dict[str, Any]] = []
        n_samples = len(x)

        dates_list = dates.tolist() if hasattr(dates, "tolist") else list(dates)

        min_required = self.train_window + self.test_window
        if n_samples < min_required:
            raise ValueError(
                "Insufficient data for walk-forward analysis. "
                f"Need at least {min_required} samples "
                f"(train_window={self.train_window} + test_window={self.test_window}), "
                f"but only have {n_samples} samples."
            )

        start_idx = 0
        end_train_idx = self.train_window
        fold_num = 1

        while end_train_idx < n_samples:
            end_test_idx = min(end_train_idx + self.test_window, n_samples)
            if end_test_idx <= end_train_idx:
                break

            train_indices = list(range(start_idx, end_train_idx))
            test_indices = list(range(end_train_idx, end_test_idx))

            splits.append(
                {
                    "fold": fold_num,
                    "train_indices": train_indices,
                    "test_indices": test_indices,
                    "train_dates": (dates_list[start_idx], dates_list[end_train_idx - 1]),
                    "test_dates": (dates_list[end_train_idx], dates_list[end_test_idx - 1]),
                }
            )

            fold_num += 1
            end_train_idx += self.retrain_frequency
            if self.window_type == "sliding":
                start_idx += self.retrain_frequency

        if not splits:
            raise ValueError(
                f"No valid folds generated. Data size: {n_samples}, "
                f"Train window: {self.train_window}, Test window: {self.test_window}"
            )

        return splits

    # ------------------------------------------------------------------
    # Per-fold metric helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Fraction of predictions whose sign matches the actual sign."""
        return float(np.mean(np.sign(y_true) == np.sign(y_pred)) * 100)

    @staticmethod
    def _sharpe_ratio(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Annualised Sharpe ratio of a long/short strategy driven by predictions.

        A position of +1 is taken when the prediction is positive, -1 otherwise.
        The strategy return for each period is ``sign(prediction) * actual_return``.
        """
        signs = np.where(y_pred >= 0, 1.0, -1.0)
        strategy_returns = signs * y_true
        std = np.std(strategy_returns)
        if std == 0.0:
            return 0.0
        return float(np.mean(strategy_returns) / std * _ANNUALISATION_FACTOR)

    @staticmethod
    def _flag_degradation(fold_metrics: list[dict[str, Any]], metric: str = "r2") -> bool:
        """Return ``True`` when the chosen metric deteriorates materially across folds.

        A model is flagged when the value of *metric* in the last fold drops by
        more than ``_DEGRADATION_R2_DROP_THRESHOLD`` relative to the first fold's
        absolute value (or relative to 1 when the first-fold value is near zero).
        """
        if len(fold_metrics) < 2:
            return False
        first_val = fold_metrics[0].get(metric, 0.0)
        last_val = fold_metrics[-1].get(metric, 0.0)
        reference = abs(first_val) if abs(first_val) > 1e-6 else 1.0
        drop = (first_val - last_val) / reference
        return drop > _DEGRADATION_R2_DROP_THRESHOLD

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(
        self,
        x: pd.DataFrame,
        y: pd.Series,
        dates: Any,
        prices: pd.Series,
        model_factory: Callable,
        *,
        predictor_params: dict | None = None,
    ) -> dict[str, Any]:
        """Perform walk-forward validation using *model_factory* to build each fold's model.

        *model_factory* must be a callable with the signature::

            model_factory(params) -> predictor

        where the returned *predictor* exposes ``train(x_train, y_train)`` and
        ``predict(x_test)`` methods.  Passing ``XGBoostStockPredictor`` directly
        is therefore valid.

        Returns a dictionary containing:
            * ``predictions`` / ``actuals`` — concatenated arrays over all folds
            * ``dates`` / ``prices`` — matching arrays for the test windows
            * ``fold_boundaries`` — per-fold date metadata
            * ``fold_metrics`` — per-fold R², RMSE, MAE, directional accuracy, Sharpe
            * ``metrics`` — aggregate metrics across all folds
            * ``regime_sensitive`` — ``True`` when degradation is detected
        """
        logger.info("=" * 60)
        logger.info("Walk-Forward Analysis (%s window)", self.window_type)
        logger.info("=" * 60)
        logger.info("Train Window: %d rows (~%.1f years)", self.train_window, self.train_window / 252)
        logger.info("Test Window:  %d rows", self.test_window)
        logger.info("Retrain Freq: %d rows", self.retrain_frequency)

        splits = self.split(x, dates)
        logger.info("Number of folds: %d", len(splits))

        all_predictions: list[float] = []
        all_actuals: list[float] = []
        all_dates: list[Any] = []
        all_prices: list[float] = []
        fold_boundaries: list[dict[str, Any]] = []
        per_fold_metrics: list[dict[str, Any]] = []

        dates_list = dates.tolist() if hasattr(dates, "tolist") else list(dates)

        for split in splits:
            fold = split["fold"]
            train_idx = split["train_indices"]
            test_idx = split["test_indices"]

            logger.info(
                "Fold %d/%d  train=%d rows  test=%d rows",
                fold,
                len(splits),
                len(train_idx),
                len(test_idx),
            )

            x_train = x.iloc[train_idx]
            y_train = y.iloc[train_idx]
            x_test = x.iloc[test_idx]
            y_test = y.iloc[test_idx]

            predictor = model_factory(predictor_params)
            predictor.train(x_train, y_train)
            predictions = predictor.predict(x_test)

            y_true_arr = np.asarray(y_test)
            y_pred_arr = np.asarray(predictions)

            fold_rmse = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))
            fold_mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
            fold_r2 = float(r2_score(y_true_arr, y_pred_arr))
            fold_dir_acc = self._directional_accuracy(y_true_arr, y_pred_arr)
            fold_sharpe = self._sharpe_ratio(y_true_arr, y_pred_arr)

            logger.info(
                "  R²=%.4f  RMSE=%.6f  MAE=%.6f  DirAcc=%.1f%%  Sharpe=%.3f",
                fold_r2,
                fold_rmse,
                fold_mae,
                fold_dir_acc,
                fold_sharpe,
            )

            all_predictions.extend(y_pred_arr.tolist())
            all_actuals.extend(y_true_arr.tolist())
            all_dates.extend([dates_list[i] for i in test_idx])
            all_prices.extend(prices.iloc[test_idx].values.tolist())

            fold_boundaries.append(
                {
                    "fold": fold,
                    "start_date": split["test_dates"][0],
                    "end_date": split["test_dates"][1],
                    "retrain_date": split["train_dates"][1],
                }
            )

            fold_entry = {
                "fold": fold,
                "r2": fold_r2,
                "rmse": fold_rmse,
                "mae": fold_mae,
                "directional_accuracy": fold_dir_acc,
                "sharpe": fold_sharpe,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
            }
            per_fold_metrics.append(fold_entry)
            self.fold_results.append(fold_entry)

        overall_actuals = np.array(all_actuals)
        overall_preds = np.array(all_predictions)
        overall_rmse = float(np.sqrt(mean_squared_error(overall_actuals, overall_preds)))
        overall_mae = float(mean_absolute_error(overall_actuals, overall_preds))
        overall_r2 = float(r2_score(overall_actuals, overall_preds))
        overall_dir_acc = self._directional_accuracy(overall_actuals, overall_preds)
        overall_sharpe = self._sharpe_ratio(overall_actuals, overall_preds)

        regime_sensitive = self._flag_degradation(per_fold_metrics, metric="r2")

        logger.info("=" * 60)
        logger.info("Walk-Forward Overall Performance:")
        logger.info("  R²:            %.8f", overall_r2)
        logger.info("  RMSE:          %.8f", overall_rmse)
        logger.info("  MAE:           %.8f", overall_mae)
        logger.info("  Dir. Accuracy: %.2f%%", overall_dir_acc)
        logger.info("  Sharpe:        %.4f", overall_sharpe)
        if regime_sensitive:
            logger.warning(
                "\u26a0  Model shows significant performance degradation across folds (regime sensitivity detected)"
            )
        logger.info("=" * 60)

        return {
            "predictions": overall_preds,
            "actuals": overall_actuals,
            "dates": all_dates,
            "prices": np.array(all_prices),
            "fold_boundaries": fold_boundaries,
            "fold_metrics": per_fold_metrics,
            "metrics": {
                "rmse": overall_rmse,
                "mae": overall_mae,
                "r2": overall_r2,
                "mse": float(overall_rmse**2),
                "directional_accuracy": overall_dir_acc,
                "sharpe": overall_sharpe,
            },
            "regime_sensitive": regime_sensitive,
        }


def do_walk_forward_validation(
    x, dates, prices, use_walk_forward: bool, y
) -> tuple[bool, dict[str, np.ndarray[Any, np.dtype[Any]] | list[Any] | dict[str, float | Any]]]:
    logger.info("=" * 60)
    logger.info("Using Walk-Forward Analysis")
    logger.info("=" * 60)

    wf_results: dict[str, Any] = {}
    try:
        # Initialize walk-forward validator
        wf_validator = WalkForwardValidator(
            train_window=WALK_FORWARD_TRAIN_WINDOW,
            test_window=WALK_FORWARD_TEST_WINDOW,
            retrain_frequency=WALK_FORWARD_RETRAIN_FREQ,
        )

        # Perform walk-forward validation
        wf_results = wf_validator.validate(
            x,
            y,
            dates,
            prices,
            XGBoostStockPredictor,
        )
    except ValueError as e:
        logger.warning("Walk-forward analysis failed: %s", e)
        logger.warning("Falling back to simple train/test split")
        use_walk_forward = False

    return use_walk_forward, wf_results


def second_pass_walk_forward(
    x,
    symbol,
    wf_results: dict[str, np.ndarray[Any, np.dtype[Any]] | list[Any] | dict[str, float | Any]],
    y,
    visualisation: bool = False,
) -> tuple[
    XGBoostStockPredictor,
    dict[str, np.ndarray[Any, np.dtype[Any]] | list[Any] | dict[str, float | Any] | float | Any],
    np.ndarray[Any, np.dtype[Any]] | list[Any] | dict[str, float | Any],
    Series,
    Series,
]:
    # Extract results for compatibility with existing code
    test_results = {
        "predictions": wf_results["predictions"],
        "rmse": wf_results["metrics"]["rmse"],
        "mae": wf_results["metrics"]["mae"],
        "r2": wf_results["metrics"]["r2"],
        "mse": wf_results["metrics"]["mse"],
    }

    y_test = pd.Series(wf_results["actuals"], index=wf_results["dates"])
    dates_test = wf_results["dates"]
    prices_test = pd.Series(wf_results["prices"], index=wf_results["dates"])

    # Create a final model for feature importance (trained on all data)
    logger.info("Training final model on all data for feature importance...")
    predictor = XGBoostStockPredictor()
    split_idx = int(len(x) * 0.9)
    predictor.train(x.iloc[:split_idx], y.iloc[:split_idx], x.iloc[split_idx:], y.iloc[split_idx:])

    if visualisation:
        generate_walk_forward_plots(wf_results, symbol)

    return dates_test, predictor, prices_test, test_results, y_test
