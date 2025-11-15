import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler


class XGBoostStockPredictor:
    """
    XGBoost model for stock price prediction
    """

    def __init__(self, params=None):
        if params is None:
            # THIS PRODUCES 4 SUCCESSFUL MODELS
            # tuned_params = {
            #     "objective": "reg:squarederror",
            #     "learning_rate": 0.05,  # <-- lower LR for smoother learning
            #     "max_depth": 5,  # slightly deeper to capture nonlinearity
            #     "min_child_weight": 4,
            #     "subsample": 0.9,
            #     "colsample_bytree": 0.8,
            #     "lambda": 1.2,
            #     "alpha": 0.6,
            #     "gamma": 0.1,
            #     "n_estimators": 1500,  # <-- more trees with smaller steps
            #     "tree_method": "exact",
            #     "seed": 42,
            # }

            params = {
                "objective": "reg:squarederror",
                "learning_rate": 0.04,  # <-- lower LR for smoother learning
                "max_depth": 4,  # slightly deeper to capture nonlinearity
                "min_child_weight": 1,
                "subsample": 0.7,
                "colsample_bytree": 0.7,
                "lambda": 0.6,
                "alpha": 0.3,
                "gamma": 0.0,
                "n_estimators": 500,  # <-- more trees with smaller steps
                "tree_method": "exact",
                "seed": 42,
            }

        self.params = params
        self.model = None
        self.feature_importance = None
        self.feature_columns = None
        self.scaler = StandardScaler()

    def train(self, x_train, y_train, x_val=None, y_val=None):
        """Train the XGBoost model"""
        print("\n" + "=" * 60)
        print("Training XGBoost Model")
        print("=" * 60)

        # Store feature columns for later use during prediction
        if isinstance(x_train, pd.DataFrame):
            self.feature_columns = list(x_train.columns)
            print(f"Stored {len(self.feature_columns)} feature columns")

        # Scale features
        print("Scaling features...")
        x_train_scaled = self.scaler.fit_transform(x_train)
        x_train_scaled = pd.DataFrame(x_train_scaled, columns=x_train.columns, index=x_train.index)

        # Extract early stopping rounds if present
        early_stopping = self.params.pop("early_stopping_rounds", None)

        self.model = xgboost.XGBRegressor(**self.params)

        if x_val is not None and y_val is not None and early_stopping is not None:
            x_val_scaled = self.scaler.transform(x_val)
            x_val_scaled = pd.DataFrame(x_val_scaled, columns=x_val.columns, index=x_val.index)

            eval_set = [(x_train_scaled, y_train), (x_val_scaled, y_val)]
            self.model.fit(x_train_scaled, y_train, eval_set=eval_set, verbose=False)
        else:
            self.model.fit(x_train_scaled, y_train)

        # Store feature importance
        self.feature_importance = pd.DataFrame(
            {"feature": x_train.columns, "importance": self.model.feature_importances_}
        ).sort_values("importance", ascending=False)

        print("✓ Model trained successfully")
        print(f"✓ Number of trees: {self.model.n_estimators}")
        print(f"✓ Best iteration: {self.model.best_iteration if hasattr(self.model, 'best_iteration') else 'N/A'}")
        print(f"✓ Max depth: {self.params['max_depth']}")

    def predict(self, x):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet")

        # Ensure features are in the correct order
        if self.feature_columns is not None and isinstance(x, pd.DataFrame):
            x = x[self.feature_columns]

        # Scale features before prediction
        x_scaled = self.scaler.transform(x)
        x_scaled = pd.DataFrame(x_scaled, columns=x.columns, index=x.index)

        return self.model.predict(x_scaled)

    def evaluate(self, x, y, dataset_name="Test"):
        """Evaluate model performance"""
        predictions = self.predict(x)

        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, predictions)
        r2 = r2_score(y, predictions)

        # Ensure features are in the correct order
        if self.feature_columns is not None and isinstance(x, pd.DataFrame):
            x = x[self.feature_columns]

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
            "trained_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        filename = f"{save_path}/{symbol}_xgboost_model.pkl"
        with open(filename, "wb") as f:
            pickle.dump(model_data, f)

        print(f"\n✓ Model saved to: {filename}")
        if self.feature_columns:
            print(f"✓ Saved {len(self.feature_columns)} feature columns")
        return filename

    @classmethod
    def load_model(cls, symbol, load_path="models"):
        """Load a trained model and scaler"""
        filename = f"{load_path}/{symbol}_xgboost_model.pkl"

        if not os.path.exists(filename):
            raise FileNotFoundError(f"Model file not found: {filename}")

        with open(filename, "rb") as f:
            model_data = pickle.load(f)

        # Create instance and restore model
        instance = cls(params=model_data["params"])
        instance.model = model_data["model"]
        instance.scaler = model_data["scaler"]
        instance.feature_importance = model_data["feature_importance"]
        instance.feature_columns = model_data.get("feature_columns")

        print(f"\n✓ Model loaded from: {filename}")
        print(f"  Trained on: {model_data['trained_date']}")
        if instance.feature_columns:
            print(f"  Features: {len(instance.feature_columns)} columns stored")

        return instance
