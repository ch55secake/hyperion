import os
import pickle
from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from src.util import logger


class Model(ABC):
    def __init__(self, model_name: str, params: dict | None = None):
        self.model_name = model_name
        self.params: dict = params if params is not None else {}
        self.model = None
        self.feature_importance = None
        self.feature_columns = None
        self.scaler = StandardScaler()
        self.numeric_columns = None
        self.categorical_columns = None

    @abstractmethod
    def train(self, x_train, y_train, x_val=None, y_val=None):
        pass

    @abstractmethod
    def predict(self, x):
        pass

    def evaluate(self, x, y, dataset_name="Test"):
        """Evaluate model performance"""
        predictions = self.predict(x)

        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, predictions)
        r2 = r2_score(y, predictions)

        logger.info(f"{dataset_name} Set Performance:")
        logger.info(f"  MSE:  {mse:.8f}")
        logger.info(f"  RMSE: {rmse:.8f}")
        logger.info(f"  MAE:  {mae:.8f}")
        logger.info(f"  R\u00b2:   {r2:.8f}")

        return {"predictions": predictions, "mse": mse, "rmse": rmse, "mae": mae, "r2": r2}

    def save_model(self, symbol, save_path="models"):
        """Save the trained model and scaler"""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")

        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "params": self.params,
            "feature_importance": self.feature_importance,
            "feature_columns": self.feature_columns,
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "trained_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        os.makedirs(save_path, exist_ok=True)
        filename = f"{save_path}/{symbol}_{self.model_name}_model.pkl"
        with open(filename, "wb") as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to: {filename}")
        return filename

    @classmethod
    def load_model(cls, model_name: str, symbol: str, load_path: str = "models"):
        """Load a trained model and scaler"""
        filename = f"{load_path}/{symbol}_{model_name}_model.pkl"

        if not os.path.exists(filename):
            raise FileNotFoundError(f"Model file not found: {filename}")

        with open(filename, "rb") as f:
            model_data = pickle.load(f)

        instance = cls(model_name=model_data.get("model_name", "unknown"), params=model_data["params"])
        instance.model = model_data["model"]
        instance.scaler = model_data["scaler"]
        instance.feature_importance = model_data["feature_importance"]
        instance.feature_columns = model_data.get("feature_columns")
        instance.numeric_columns = model_data.get("numeric_columns")
        instance.categorical_columns = model_data.get("categorical_columns")

        logger.info(f"Model loaded from: {filename}")
        logger.info(f"  Trained on: {model_data['trained_date']}")
        if instance.feature_columns:
            logger.debug(f"  Features: {len(instance.feature_columns)} columns stored")
        if instance.categorical_columns:
            logger.debug(f"  Categorical features: {instance.categorical_columns}")

        return instance

    def _prepare_columns(self, x_train):
        if isinstance(x_train, pd.DataFrame):
            self.feature_columns = list(x_train.columns)
            logger.debug(f"Stored {len(self.feature_columns)} feature columns")

        self.categorical_columns = x_train.select_dtypes(include=["category"]).columns.tolist()
        self.numeric_columns = x_train.select_dtypes(include=["number"]).columns.tolist()

        logger.debug(f"Numeric columns: {len(self.numeric_columns)}")
        logger.debug(f"Categorical columns: {len(self.categorical_columns)}")
        if self.categorical_columns:
            logger.debug(f"  Categories: {self.categorical_columns}")

        object_cols = x_train.select_dtypes(include=["object"]).columns.tolist()
        if object_cols:
            logger.warning(f"Found {len(object_cols)} object columns that should be categorical: {object_cols}")
            logger.info("Converting them to category dtype...")
            x_train = x_train.copy()
            for col in object_cols:
                x_train[col] = x_train[col].astype("category")
            self.categorical_columns = x_train.select_dtypes(include=["category"]).columns.tolist()
            logger.debug(f"  Updated categorical columns: {self.categorical_columns}")

        logger.debug("Scaling numeric features...")
        x_train_processed = x_train.copy()

        if self.numeric_columns:
            x_train_scaled = self.scaler.fit_transform(x_train[self.numeric_columns])
            for i, col in enumerate(self.numeric_columns):
                x_train_processed[col] = x_train_scaled[:, i]

        return x_train_processed, object_cols

    def _prepare_numeric_and_categorical_columns(self, x_val_processed, object_cols, x_val):
        for col in object_cols if object_cols else []:
            if col in x_val_processed.columns:
                x_val_processed[col] = x_val_processed[col].astype("category")

        if self.numeric_columns:
            x_val_scaled = self.scaler.transform(x_val[self.numeric_columns])
            for i, col in enumerate(self.numeric_columns):
                x_val_processed[col] = x_val_scaled[:, i]

        return x_val_processed

    def _prepare_prediction(self, x):
        if self.model is None:
            raise ValueError("Model not trained yet")

        if self.feature_columns is not None and isinstance(x, pd.DataFrame):
            x = x[self.feature_columns]

        object_cols = x.select_dtypes(include=["object"]).columns.tolist()
        if object_cols:
            x = x.copy()
            for col in object_cols:
                x[col] = x[col].astype("category")

        x_processed = x.copy()
        if self.numeric_columns:
            x_scaled = self.scaler.transform(x[self.numeric_columns])
            for i, col in enumerate(self.numeric_columns):
                x_processed[col] = x_scaled[:, i]

        return x_processed
