from typing import Any

import numpy as np
import pandas as pd
from pandas import Series
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.visualisation import generate_walk_forward_plots
from src.xbg import XGBoostStockPredictor

WALK_FORWARD_TRAIN_WINDOW = 180  # ~9 months of training data (reduced from 252)
WALK_FORWARD_TEST_WINDOW = 20  # ~1 month of testing
WALK_FORWARD_RETRAIN_FREQ = 20  # Retrain every ~1 month


class WalkForwardValidator:
    """Implements walk-forward analysis for time series"""

    def __init__(self, train_window=252, test_window=21, retrain_frequency=21):
        """
        Args:
            train_window: Number of days to use for training (252 = ~1 year)
            test_window: Number of days to test on before retraining
            retrain_frequency: How often to retrain (days)
        """
        self.train_window = train_window
        self.test_window = test_window
        self.retrain_frequency = retrain_frequency
        self.fold_results = []

    def split(self, x, dates):
        """Generate train/test splits for walk-forward"""
        splits = []
        n_samples = len(x)

        # Convert dates to list if it's an Index
        if hasattr(dates, "tolist"):
            dates_list = dates.tolist()
        else:
            dates_list = list(dates)

        # Validate we have enough data
        min_required = self.train_window + self.test_window
        if n_samples < min_required:
            raise ValueError(
                "Insufficient data for walk-forward analysis. "
                f"Need at least {min_required} samples "
                f"(train_window={self.train_window} + test_window={self.test_window}), "
                f"but only have {n_samples} samples."
            )

        # Start with initial training window
        start_idx = 0
        end_train_idx = self.train_window

        fold_num = 1
        while end_train_idx < n_samples:
            # Determine test period
            end_test_idx = min(end_train_idx + self.test_window, n_samples)

            if end_test_idx <= end_train_idx:
                break

            # Create split
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

            # Move forward by retrain_frequency
            # Expanding window: keep start_idx at 0
            # Rolling window: start_idx += self.retrain_frequency
            end_train_idx += self.retrain_frequency

        if len(splits) == 0:
            raise ValueError(
                f"No valid folds generated. Data size: {n_samples}, "
                f"Train window: {self.train_window}, Test window: {self.test_window}"
            )

        return splits

    def validate(self, x, y, dates, prices, predictor_class, predictor_params=None):
        """
        Perform walk-forward validation

        Returns:
            Dictionary with combined predictions and fold information
        """
        print("\n" + "=" * 60)
        print("Walk-Forward Analysis")
        print("=" * 60)
        print(f"Train Window: {self.train_window} days (~{self.train_window / 252:.1f} years)")
        print(f"Test Window: {self.test_window} days")
        print(f"Retrain Frequency: {self.retrain_frequency} days")

        splits = self.split(x, dates)
        print(f"Number of folds: {len(splits)}")

        all_predictions = []
        all_actuals = []
        all_dates = []
        all_prices = []
        fold_boundaries = []

        for split in splits:
            fold = split["fold"]
            train_idx = split["train_indices"]
            test_idx = split["test_indices"]

            print(f"\nFold {fold}/{len(splits)}:")
            print(f"  Train: {split['train_dates'][0]} to {split['train_dates'][1]} ({len(train_idx)} samples)")
            print(f"  Test:  {split['test_dates'][0]} to {split['test_dates'][1]} ({len(test_idx)} samples)")

            # Get train/test data
            x_train = x.iloc[train_idx]
            y_train = y.iloc[train_idx]
            x_test = x.iloc[test_idx]
            y_test = y.iloc[test_idx]

            # Train model for this fold
            predictor = predictor_class(predictor_params)
            predictor.train(x_train, y_train)

            # Make predictions
            predictions = predictor.predict(x_test)

            # Calculate fold metrics
            fold_mse = mean_squared_error(y_test, predictions)
            fold_rmse = np.sqrt(fold_mse)
            fold_mae = mean_absolute_error(y_test, predictions)

            print(f"  Fold RMSE: {fold_rmse:.6f}")
            print(f"  Fold MAE:  {fold_mae:.6f}")

            # Store results
            all_predictions.extend(predictions)
            all_actuals.extend(y_test.values)

            # Handle dates indexing
            if hasattr(dates, "tolist"):
                test_dates = [dates[i] for i in test_idx]
            else:
                test_dates = [dates[i] for i in test_idx]
            all_dates.extend(test_dates)
            all_prices.extend(prices.iloc[test_idx].values)

            fold_boundaries.append(
                {
                    "fold": fold,
                    "start_date": split["test_dates"][0],
                    "end_date": split["test_dates"][1],
                    "retrain_date": split["train_dates"][1],
                }
            )

            self.fold_results.append(
                {"fold": fold, "rmse": fold_rmse, "mae": fold_mae, "n_train": len(train_idx), "n_test": len(test_idx)}
            )

        # Calculate overall metrics
        overall_mse = mean_squared_error(all_actuals, all_predictions)
        overall_rmse = np.sqrt(overall_mse)
        overall_mae = mean_absolute_error(all_actuals, all_predictions)
        overall_r2 = r2_score(all_actuals, all_predictions)

        print(f"\n{'=' * 60}")
        print("Walk-Forward Overall Performance:")
        print(f"  RMSE: {overall_rmse:.8f}")
        print(f"  MAE:  {overall_mae:.8f}")
        print(f"  R²:   {overall_r2:.8f}")
        print(f"{'=' * 60}")

        return {
            "predictions": np.array(all_predictions),
            "actuals": np.array(all_actuals),
            "dates": all_dates,
            "prices": np.array(all_prices),
            "fold_boundaries": fold_boundaries,
            "metrics": {"rmse": overall_rmse, "mae": overall_mae, "r2": overall_r2, "mse": overall_mse},
        }


def do_walk_forward_validation(
    x, dates, prices, use_walk_forward: bool, y
) -> tuple[bool, dict[str, np.ndarray[Any, np.dtype[Any]] | list[Any] | dict[str, float | Any]]]:
    print("\n" + "=" * 60)
    print("Using Walk-Forward Analysis")
    print("=" * 60)

    try:
        # Initialize walk-forward validator
        wf_validator = WalkForwardValidator(
            train_window=WALK_FORWARD_TRAIN_WINDOW,
            test_window=WALK_FORWARD_TEST_WINDOW,
            retrain_frequency=WALK_FORWARD_RETRAIN_FREQ,
        )

        # Perform walk-forward validation
        wf_results = wf_validator.validate(
            x, y, dates, prices, XGBoostStockPredictor, predictor_params=None  # Uses default params
        )
    except ValueError as e:
        print(f"\n⚠️  Walk-forward analysis failed: {e}")
        print("   Falling back to simple train/test split")
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
    print("\nTraining final model on all data for feature importance...")
    predictor = XGBoostStockPredictor()
    split_idx = int(len(x) * 0.9)
    predictor.train(x.iloc[:split_idx], y.iloc[:split_idx], x.iloc[split_idx:], y.iloc[split_idx:])

    if visualisation:
        generate_walk_forward_plots(wf_results, symbol)

    return dates_test, predictor, prices_test, test_results, y_test
