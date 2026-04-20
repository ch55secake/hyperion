import pandas as pd
import lightgbm as lgb

from ..model import Model
from src.util import logger, get_device


class LightGBMStockPredictor(Model):
    """
    LightGBM model for stock price prediction with categorical feature support
    """

    def __init__(self, params: dict | None = None):
        if params is None:
            params = {
                "objective": "regression",
                "metric": "rmse",
                "verbosity": -1,
                "device": "gpu" if get_device() == "cuda" else "cpu",
                "max_depth": -1,
                "min_data_in_leaf": 5,
                "min_gain_to_split": 0.0,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 1,
                "n_estimators": 500,
                "early_stopping_rounds": 50,
                "seed": 42,
            }

        super().__init__("lightgbm", params=params)

    def train(self, x_train, y_train, x_val=None, y_val=None):
        """Train the LightGBM model"""
        logger.info("=" * 60)
        logger.info("Training LightGBM Model")
        logger.info("=" * 60)

        x_train_processed, object_cols = self._prepare_columns(x_train)

        train_data = lgb.Dataset(
            x_train_processed,
            label=y_train,
            categorical_feature=self.categorical_columns if self.categorical_columns else "auto",
        )

        early_stopping_rounds = self.params.get("early_stopping_rounds")
        train_params = {k: v for k, v in self.params.items() if k != "early_stopping_rounds"}

        valid_sets = [train_data]
        valid_names = ["train"]

        if x_val is not None and y_val is not None:
            x_val_processed = self._prepare_numeric_and_categorical_columns(x_val.copy(), object_cols, x_val)

            val_data = lgb.Dataset(
                x_val_processed,
                label=y_val,
                categorical_feature=self.categorical_columns if self.categorical_columns else "auto",
            )
            valid_sets.append(val_data)
            valid_names.append("valid")
            logger.debug(f"x_val_processed shape: {x_val_processed.shape}")
            logger.debug(f"y_val shape: {y_val.shape}")
            logger.debug(f"x_val_processed length: {len(x_val_processed)}")
            logger.debug(f"y_val length: {len(y_val)}")

        callbacks = []
        if early_stopping_rounds is not None and x_val is not None and y_val is not None:
            callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False))
            logger.info("Early stopping enabled with %d rounds", early_stopping_rounds)

        self.model = lgb.train(
            train_params,
            train_data,
            valid_sets=valid_sets,
            valid_names=valid_names,
            num_boost_round=self.params.get("n_estimators", 1500),
            callbacks=callbacks or None,
        )

        feature_names = x_train_processed.columns.tolist()
        feature_importances = self.model.feature_importance()

        if len(feature_names) != len(feature_importances):
            logger.warning(
                f"Feature name count ({len(feature_names)}) "
                + f"doesn't match importance count ({len(feature_importances)})"
            )
            feature_names = [f"feature_{i}" for i in range(len(feature_importances))]

        self.feature_importance = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": feature_importances,
            }
        ).sort_values("importance", ascending=False)

        logger.info("Model trained successfully")
        logger.info(f"Number of trees: {self.model.best_iteration or self.params['n_estimators']}")
        logger.info(f"Max depth: {self.params['max_depth']}")

        logger.debug("Top 10 Most Important Features:")
        logger.debug(self.feature_importance.head(10).to_string(index=False))

    def predict(self, x):
        """Make predictions"""
        return self.model.predict(self._prepare_prediction(x), num_iteration=self.model.best_iteration)  # ty: ignore[unresolved-attribute]
