import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler


class LightGBMStockPredictor:
    """
    LightGBM model for stock price prediction with categorical feature support
    """

    def __init__(self, params=None):
        if params is None:
            params = {
                "objective": "regression",
                "metric": "rmse",
                "verbosity": -1,
                "max_depth": -1,
                "min_data_in_leaf": 5,
                "min_gain_to_split": 0.0,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 1,
                "n_estimators": 500,
                "seed": 42,
            }

        self.params = params
        self.model = None
        self.feature_importance = None
        self.feature_columns = None
        self.scaler = StandardScaler()
        self.numeric_columns = None
        self.categorical_columns = None

    def train(self, x_train, y_train, x_val=None, y_val=None):
        """Train the LightGBM model"""
        print("\n" + "=" * 60)
        print("Training LightGBM Model")
        print("=" * 60)

        if isinstance(x_train, pd.DataFrame):
            self.feature_columns = list(x_train.columns)
            print(f"Stored {len(self.feature_columns)} feature columns")

        self.categorical_columns = x_train.select_dtypes(include=["category"]).columns.tolist()
        self.numeric_columns = x_train.select_dtypes(include=["number"]).columns.tolist()

        print(f"Numeric columns: {len(self.numeric_columns)}")
        print(f"Categorical columns: {len(self.categorical_columns)}")
        if self.categorical_columns:
            print(f"  Categories: {self.categorical_columns}")

        object_cols = x_train.select_dtypes(include=["object"]).columns.tolist()
        if object_cols:
            print(f"⚠️ Warning: Found {len(object_cols)} object columns that should be categorical: {object_cols}")
            print("Converting them to category dtype...")
            x_train = x_train.copy()
            for col in object_cols:
                x_train[col] = x_train[col].astype("category")
            self.categorical_columns = x_train.select_dtypes(include=["category"]).columns.tolist()
            print(f"  Updated categorical columns: {self.categorical_columns}")

        print("Scaling numeric features...")
        x_train_processed = x_train.copy()

        if self.numeric_columns:
            x_train_scaled = self.scaler.fit_transform(x_train[self.numeric_columns])
            for i, col in enumerate(self.numeric_columns):
                x_train_processed[col] = x_train_scaled[:, i]

        train_data = lgb.Dataset(
            x_train_processed,
            label=y_train,
            categorical_feature=self.categorical_columns if self.categorical_columns else "auto",
        )

        valid_sets = [train_data]
        valid_names = ["train"]

        if x_val is not None and y_val is not None:
            x_val_processed = x_val.copy()

            for col in object_cols if object_cols else []:
                if col in x_val_processed.columns:
                    x_val_processed[col] = x_val_processed[col].astype("category")

            if self.numeric_columns:
                x_val_scaled = self.scaler.transform(x_val[self.numeric_columns])
                for i, col in enumerate(self.numeric_columns):
                    x_val_processed[col] = x_val_scaled[:, i]

            val_data = lgb.Dataset(
                x_val_processed,
                label=y_val,
                categorical_feature=self.categorical_columns if self.categorical_columns else "auto",
            )
            valid_sets.append(val_data)
            valid_names.append("valid")

        self.model = lgb.train(
            self.params,
            train_data,
            valid_sets=valid_sets,
            valid_names=valid_names,
            num_boost_round=self.params.get("n_estimators", 1500),
        )

        feature_names = x_train_processed.columns.tolist()
        feature_importances = self.model.feature_importance()

        if len(feature_names) != len(feature_importances):
            print(
                f"⚠️ Warning: Feature name count ({len(feature_names)}) doesn't match importance count ({len(feature_importances)})"
            )
            feature_names = [f"feature_{i}" for i in range(len(feature_importances))]

        self.feature_importance = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": feature_importances,
            }
        ).sort_values("importance", ascending=False)

        print("✓ Model trained successfully")
        print(f"✓ Number of trees: {self.model.best_iteration or self.params['n_estimators']}")
        print(f"✓ Max depth: {self.params['max_depth']}")

        print("\nTop 10 Most Important Features:")
        print(self.feature_importance.head(10).to_string(index=False))

    def predict(self, x):
        """Make predictions"""
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

        return self.model.predict(x_processed, num_iteration=self.model.best_iteration)

    def evaluate(self, x, y, dataset_name="Test"):
        """Evaluate model performance"""
        predictions = self.predict(x)

        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, predictions)
        r2 = r2_score(y, predictions)

        print(f"\n{dataset_name} Set Performance:")
        print(f"  MSE:  {mse:.8f}")
        print(f"  RMSE: {rmse:.8f}")
        print(f"  MAE:  {mae:.8f}")
        print(f"  R²:   {r2:.8f}")

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
        filename = f"{save_path}/{symbol}_lightgbm_model.pkl"
        with open(filename, "wb") as f:
            pickle.dump(model_data, f)

        print(f"\n✓ Model saved to: {filename}")
        return filename

    @classmethod
    def load_model(cls, symbol, load_path="models"):
        """Load a trained model and scaler"""
        filename = f"{load_path}/{symbol}_lightgbm_model.pkl"

        if not os.path.exists(filename):
            raise FileNotFoundError(f"Model file not found: {filename}")

        with open(filename, "rb") as f:
            model_data = pickle.load(f)

        instance = cls(params=model_data["params"])
        instance.model = model_data["model"]
        instance.scaler = model_data["scaler"]
        instance.feature_importance = model_data["feature_importance"]
        instance.feature_columns = model_data.get("feature_columns")
        instance.numeric_columns = model_data.get("numeric_columns")
        instance.categorical_columns = model_data.get("categorical_columns")

        print(f"\n✓ Model loaded from: {filename}")
        print(f"  Trained on: {model_data['trained_date']}")
        if instance.feature_columns:
            print(f"  Features: {len(instance.feature_columns)} columns stored")
        if instance.categorical_columns:
            print(f"  Categorical features: {instance.categorical_columns}")

        return instance
