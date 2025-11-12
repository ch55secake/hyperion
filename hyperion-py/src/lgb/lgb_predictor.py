import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler


class LightGBMStockPredictor:
    """LightGBM model for stock price prediction"""

    def __init__(self, params=None):
        if params is None:
            # tuned_params_v2 = {
            #     "objective": "regression",
            #     "metric": "rmse",
            #     "learning_rate": 0.04,
            #     "num_leaves": 64,
            #     "max_depth": 7,
            #     "min_data_in_leaf": 20,
            #     "feature_fraction": 0.8,   # equivalent to colsample_bytree
            #     "bagging_fraction": 0.9,   # equivalent to subsample
            #     "bagging_freq": 1,
            #     "lambda_l1": 0.8,          # equivalent to alpha
            #     "lambda_l2": 1.5,          # equivalent to lambda
            #     "n_estimators": 1500,
            #     # "verbosity": -1,
            #     # "random_state": 42,
            #     "device_type": "cpu",      # set to "cpu" if no GPU
            # }
            # params = tuned_params_v2
            params = {
                "objective": "regression",
                "metric": "rmse",
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
        self.scaler = StandardScaler()

    def train(self, x_train, y_train, x_val=None, y_val=None):
        """Train the LightGBM model"""
        print("\n" + "=" * 60)
        print("Training LightGBM Model")
        print("=" * 60)

        # Scale features
        print("Scaling features...")
        x_train_scaled = self.scaler.fit_transform(x_train)
        x_train_scaled = pd.DataFrame(x_train_scaled, columns=x_train.columns, index=x_train.index)

        # Create dataset for LightGBM
        train_data = lgb.Dataset(x_train_scaled, label=y_train)

        valid_sets = [train_data]
        valid_names = ["train"]

        if x_val is not None and y_val is not None:
            x_val_scaled = self.scaler.transform(x_val)
            x_val_scaled = pd.DataFrame(x_val_scaled, columns=x_val.columns, index=x_val.index)
            val_data = lgb.Dataset(x_val_scaled, label=y_val)
            valid_sets.append(val_data)
            valid_names.append("valid")

        # Train with early stopping if validation data provided
        self.model = lgb.train(
            self.params,
            train_data,
            valid_sets=valid_sets,
            valid_names=valid_names,
            num_boost_round=self.params.get("n_estimators", 1500),
            # early_stopping_rounds=50 if len(valid_sets) > 1 else None,
            # verbose_eval=False,
        )

        # Store feature importance
        self.feature_importance = pd.DataFrame(
            {
                "feature": x_train.columns,
                "importance": self.model.feature_importance(),
            }
        ).sort_values("importance", ascending=False)

        print("✓ Model trained successfully")
        print(f"✓ Number of trees: {self.model.best_iteration or self.params['n_estimators']}")
        print(f"✓ Max depth: {self.params['max_depth']}")

    def predict(self, x):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet")

        # Scale features before prediction
        x_scaled = self.scaler.transform(x)
        x_scaled = pd.DataFrame(x_scaled, columns=x.columns, index=x.index)

        return self.model.predict(x_scaled, num_iteration=self.model.best_iteration)

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
            "trained_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

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

        print(f"\n✓ Model loaded from: {filename}")
        print(f"  Trained on: {model_data['trained_date']}")

        return instance
