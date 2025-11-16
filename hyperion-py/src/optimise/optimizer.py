import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
)
import xgboost as xgb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import logging
from typing import Dict, Any, Tuple
import warnings

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockModelOptimizer:
    """Optimize XGBoost and LightGBM models using Optuna with categorical feature support"""

    def __init__(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_val: pd.DataFrame,
        y_val: pd.Series,
        n_trials: int = 1000,
        n_jobs: int = -1,
        random_state: int = 42,
    ):
        """
        Initialize optimizer

        Args:
            x_train: Training features
            y_train: Training target
            x_val: Validation features
            y_val: Validation target
            n_trials: Number of optimization trials
            n_jobs: Number of parallel jobs
            random_state: Random seed
        """
        # Process features before storing
        self.X_train = self._process_features(x_train.copy())
        self.X_val = self._process_features(x_val.copy())
        self.y_train = y_train
        self.y_val = y_val
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.random_state = random_state

        # Identify column types
        self.categorical_columns = self.X_train.select_dtypes(include=["category"]).columns.tolist()
        self.numeric_columns = self.X_train.select_dtypes(include=["number"]).columns.tolist()

        logger.info(f"Optimizer initialized with {len(self.X_train.columns)} features")
        logger.info(f"  Numeric columns: {len(self.numeric_columns)}")
        logger.info(f"  Categorical columns: {len(self.categorical_columns)}")
        if self.categorical_columns:
            logger.info(f"    Categories: {self.categorical_columns}")

        # Initialize scaler for numeric features
        self.scaler = StandardScaler()

        # Scale numeric features
        if self.numeric_columns:
            X_train_scaled = self.scaler.fit_transform(self.X_train[self.numeric_columns])
            X_val_scaled = self.scaler.transform(self.X_val[self.numeric_columns])

            # Replace numeric columns with scaled values while preserving categorical
            for i, col in enumerate(self.numeric_columns):
                self.X_train[col] = X_train_scaled[:, i]
                self.X_val[col] = X_val_scaled[:, i]

        self.best_xgb_params = None
        self.best_lgb_params = None
        self.xgb_study = None
        self.lgb_study = None

    def _process_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process features to ensure categorical columns are properly typed

        Args:
            df: Input dataframe

        Returns:
            Processed dataframe with proper dtypes
        """
        # Convert object columns to category
        object_cols = df.select_dtypes(include=["object"]).columns.tolist()
        if object_cols:
            logger.info(f"Converting {len(object_cols)} object columns to category: {object_cols}")
            for col in object_cols:
                df[col] = df[col].astype("category")

        return df

    def xgboost_objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for XGBoost optimization

        Args:
            trial: Optuna trial object

        Returns:
            Validation RMSE (lower is better)
        """
        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "tree_method": "hist",  # Required for categorical support
            "verbosity": 0,
            "enable_categorical": True,  # Enable categorical support
            "seed": self.random_state,
            # Learning parameters
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 50),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
            # Regularization
            "lambda": trial.suggest_float("lambda", 0.1, 10.0, log=True),
            "alpha": trial.suggest_float("alpha", 0.0, 5.0),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            # Sampling
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
            # Other
            "max_delta_step": trial.suggest_float("max_delta_step", 0, 5),
        }

        n_estimators = trial.suggest_int("n_estimators", 100, 4000)

        model = xgb.XGBRegressor(**params, n_estimators=n_estimators)

        model.fit(
            self.X_train,
            self.y_train,
            eval_set=[(self.X_val, self.y_val)],
            early_stopping_rounds=self.n_trials,
            verbose=False,
        )

        y_pred = model.predict(self.X_val)
        rmse = np.sqrt(mean_squared_error(self.y_val, y_pred))

        # Calculate additional metrics for tracking
        mae = mean_absolute_error(self.y_val, y_pred)
        r2 = r2_score(self.y_val, y_pred)

        # Directional accuracy
        dir_acc = (np.sign(y_pred) == np.sign(self.y_val)).mean()

        # Store additional metrics
        trial.set_user_attr("mae", mae)
        trial.set_user_attr("r2", r2)
        trial.set_user_attr("directional_accuracy", dir_acc)
        trial.set_user_attr("best_iteration", model.best_iteration)

        return rmse

    def lightgbm_objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for LightGBM optimization

        Args:
            trial: Optuna trial object

        Returns:
            Validation RMSE (lower is better)
        """
        # Suggest hyperparameters
        params = {
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            "seed": self.random_state,
            "force_col_wise": True,
            # Learning parameters
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 10, 200),
            "max_depth": trial.suggest_int("max_depth", 2, 50),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-5, 100, log=True),
            # Regularization
            "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 5.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 5.0),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 1.0),
            # Sampling
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            # Other
            "max_bin": trial.suggest_int("max_bin", 128, 512),
        }

        n_estimators = trial.suggest_int("n_estimators", 100, 2000)

        # Create datasets with categorical feature support
        train_data = lgb.Dataset(
            self.X_train,
            label=self.y_train,
            categorical_feature=self.categorical_columns if self.categorical_columns else "auto",
        )
        val_data = lgb.Dataset(
            self.X_val,
            label=self.y_val,
            reference=train_data,
            categorical_feature=self.categorical_columns if self.categorical_columns else "auto",
        )

        model = lgb.train(
            params,
            train_data,
            num_boost_round=n_estimators,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=self.n_trials, verbose=False), lgb.log_evaluation(period=0)],
        )

        y_pred = model.predict(self.X_val, num_iteration=model.best_iteration)
        rmse = np.sqrt(mean_squared_error(self.y_val, y_pred))

        mae = mean_absolute_error(self.y_val, y_pred)
        r2 = r2_score(self.y_val, y_pred)
        dir_acc = (np.sign(y_pred) == np.sign(self.y_val)).mean()

        trial.set_user_attr("mae", mae)
        trial.set_user_attr("r2", r2)
        trial.set_user_attr("directional_accuracy", dir_acc)
        trial.set_user_attr("best_iteration", model.best_iteration)

        return rmse

    def optimize_xgboost(self, timeout: int = None) -> Dict[str, Any]:
        """
        Optimize XGBoost hyperparameters

        Args:
            timeout: Maximum time in seconds for optimization

        Returns:
            Dictionary with the best parameters and study
        """
        logger.info("Starting XGBoost optimization...")

        self.xgb_study = optuna.create_study(
            direction="minimize",
            study_name="xgboost_optimization",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )

        self.xgb_study.optimize(
            self.xgboost_objective, n_trials=self.n_trials, timeout=timeout, n_jobs=self.n_jobs, show_progress_bar=True
        )

        self.best_xgb_params = self.xgb_study.best_params
        best_trial = self.xgb_study.best_trial

        logger.info(f"\nXGBoost Optimization Complete!")
        logger.info(f"Best RMSE: {best_trial.value:.6f}")
        logger.info(f"Best MAE: {best_trial.user_attrs['mae']:.6f}")
        logger.info(f"Best R²: {best_trial.user_attrs['r2']:.6f}")
        logger.info(f"Best Directional Accuracy: {best_trial.user_attrs['directional_accuracy']*100:.2f}%")
        logger.info(f"Best Iteration: {best_trial.user_attrs['best_iteration']}")
        logger.info(f"\nBest Parameters:")
        for key, value in self.best_xgb_params.items():
            logger.info(f"  {key}: {value}")

        return {
            "best_params": self.best_xgb_params,
            "best_rmse": best_trial.value,
            "best_trial": best_trial,
            "study": self.xgb_study,
        }

    def optimize_lightgbm(self, timeout: int = None) -> Dict[str, Any]:
        """
        Optimize LightGBM hyperparameters

        Args:
            timeout: Maximum time in seconds for optimization

        Returns:
            Dictionary with the best parameters and study
        """
        logger.info("Starting LightGBM optimization...")

        self.lgb_study = optuna.create_study(
            direction="minimize",
            study_name="lightgbm_optimization",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )

        self.lgb_study.optimize(
            self.lightgbm_objective, n_trials=self.n_trials, timeout=timeout, n_jobs=self.n_jobs, show_progress_bar=True
        )

        self.best_lgb_params = self.lgb_study.best_params
        best_trial = self.lgb_study.best_trial

        logger.info(f"\nLightGBM Optimization Complete!")
        logger.info(f"Best RMSE: {best_trial.value:.6f}")
        logger.info(f"Best MAE: {best_trial.user_attrs['mae']:.6f}")
        logger.info(f"Best R²: {best_trial.user_attrs['r2']:.6f}")
        logger.info(f"Best Directional Accuracy: {best_trial.user_attrs['directional_accuracy']*100:.2f}%")
        logger.info(f"Best Iteration: {best_trial.user_attrs['best_iteration']}")
        logger.info(f"\nBest Parameters:")
        for key, value in self.best_lgb_params.items():
            logger.info(f"  {key}: {value}")

        return {
            "best_params": self.best_lgb_params,
            "best_rmse": best_trial.value,
            "best_trial": best_trial,
            "study": self.lgb_study,
        }

    def optimize_both(self, timeout: int = None) -> Tuple[Dict, Dict]:
        """
        Optimize both XGBoost and LightGBM

        Args:
            timeout: Maximum time in seconds for EACH optimization

        Returns:
            Tuple of (XGBoost results, LightGBM results)
        """
        xgb_results = self.optimize_xgboost(timeout=timeout)
        lgb_results = self.optimize_lightgbm(timeout=timeout)

        # Compare
        logger.info("\n" + "=" * 80)
        logger.info("COMPARISON")
        logger.info("=" * 80)
        logger.info(
            f"XGBoost  - RMSE: {xgb_results['best_rmse']:.6f}, "
            f"R²: {xgb_results['best_trial'].user_attrs['r2']:.6f}, "
            f"Dir Acc: {xgb_results['best_trial'].user_attrs['directional_accuracy']*100:.2f}%"
        )
        logger.info(
            f"LightGBM - RMSE: {lgb_results['best_rmse']:.6f}, "
            f"R²: {lgb_results['best_trial'].user_attrs['r2']:.6f}, "
            f"Dir Acc: {lgb_results['best_trial'].user_attrs['directional_accuracy']*100:.2f}%"
        )

        if xgb_results["best_rmse"] < lgb_results["best_rmse"]:
            logger.info("\n🏆 Winner: XGBoost")
        else:
            logger.info("\n🏆 Winner: LightGBM")

        return xgb_results, lgb_results

    def visualize_studies(self, save_path: str = "optuna_plots"):
        """
        Create visualization plots for optimization studies

        Args:
            save_path: Directory to save plots
        """
        import os

        os.makedirs(save_path, exist_ok=True)

        # XGBoost plots
        if self.xgb_study is not None:
            logger.info("Creating XGBoost visualization plots...")

            # Optimization history
            fig = plot_optimization_history(self.xgb_study)
            fig.write_html(f"{save_path}/xgb_optimization_history.html")

            # Parameter importances
            fig = plot_param_importances(self.xgb_study)
            fig.write_html(f"{save_path}/xgb_param_importance.html")

            # Parallel coordinate
            fig = plot_parallel_coordinate(self.xgb_study)
            fig.write_html(f"{save_path}/xgb_parallel_coordinate.html")

            logger.info(f"✓ XGBoost plots saved to {save_path}/")

        # LightGBM plots
        if self.lgb_study is not None:
            logger.info("Creating LightGBM visualization plots...")

            # Optimization history
            fig = plot_optimization_history(self.lgb_study)
            fig.write_html(f"{save_path}/lgb_optimization_history.html")

            # Parameter importances
            fig = plot_param_importances(self.lgb_study)
            fig.write_html(f"{save_path}/lgb_param_importance.html")

            # Parallel coordinate
            fig = plot_parallel_coordinate(self.lgb_study)
            fig.write_html(f"{save_path}/lgb_parallel_coordinate.html")

            logger.info(f"✓ LightGBM plots saved to {save_path}/")

    def save_results(self, filepath: str = "optuna_results.json"):
        """
        Save optimization results to file

        Args:
            filepath: Path to save results
        """
        import json

        results = {
            "xgboost": {
                "best_params": self.best_xgb_params,
                "best_rmse": self.xgb_study.best_value if self.xgb_study else None,
                "n_trials": len(self.xgb_study.trials) if self.xgb_study else 0,
            },
            "lightgbm": {
                "best_params": self.best_lgb_params,
                "best_rmse": self.lgb_study.best_value if self.lgb_study else None,
                "n_trials": len(self.lgb_study.trials) if self.lgb_study else 0,
            },
        }

        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"✓ Results saved to {filepath}")


def cross_validate_with_optuna(
    x: pd.DataFrame, y: pd.Series, model_type: str = "xgboost", n_splits: int = 5, n_trials: int = 1000
) -> Dict[str, Any]:
    """
    Perform cross-validated hyperparameter optimization

    Args:
        x: Features
        y: Target
        model_type: 'xgboost' or 'lightgbm'
        n_splits: Number of time series splits
        n_trials: Number of trials per fold

    Returns:
        Dictionary with results
    """
    logger.info(f"\nCross-Validated Optimization for {model_type.upper()}")
    logger.info("=" * 80)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(x), 1):
        logger.info(f"\nFold {fold}/{n_splits}")
        logger.info("-" * 40)

        x_train = x.iloc[train_idx]
        x_val = x.iloc[val_idx]
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        optimizer = StockModelOptimizer(
            x_train, y_train, x_val, y_val, n_trials=n_trials, n_jobs=1  # Sequential for cross-validation
        )

        if model_type.lower() == "xgboost":
            result = optimizer.optimize_xgboost()
        else:
            result = optimizer.optimize_lightgbm()

        fold_results.append(
            {
                "fold": fold,
                "best_params": result["best_params"],
                "best_rmse": result["best_rmse"],
                "best_r2": result["best_trial"].user_attrs["r2"],
                "best_dir_acc": result["best_trial"].user_attrs["directional_accuracy"],
            }
        )

    avg_rmse = np.mean([r["best_rmse"] for r in fold_results])
    avg_r2 = np.mean([r["best_r2"] for r in fold_results])
    avg_dir_acc = np.mean([r["best_dir_acc"] for r in fold_results])

    logger.info("\n" + "=" * 80)
    logger.info("CROSS-VALIDATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Average RMSE: {avg_rmse:.6f}")
    logger.info(f"Average R²: {avg_r2:.6f}")
    logger.info(f"Average Directional Accuracy: {avg_dir_acc*100:.2f}%")

    return {"fold_results": fold_results, "avg_rmse": avg_rmse, "avg_r2": avg_r2, "avg_dir_acc": avg_dir_acc}
