import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler


class XGBoostStockPredictor:
    """XGBoost model for stock price prediction"""

    def __init__(self, params=None):
        if params is None:
            params = {
                'objective': 'reg:squarederror',
                'max_depth': 50,  # Reduced from 50 - too deep causes overfitting to mean
                'learning_rate': 0.1,  # Increased from 0.05 for stronger updates
                'n_estimators': 1000,  # Reduced from 1000
                'subsample': 0.9,  # Increased from 0.8
                'colsample_bytree': 0.9,  # Increased from 0.8
                'colsample_bylevel': 0.9,  # Increased from 0.8
                'gamma': 0.0,  # Removed minimum split loss requirement
                'reg_alpha': 0.01,  # Reduced L1 regularization from 0.5
                'reg_lambda': 0.1,  # Reduced L2 regularization from 1.0
                'min_child_weight': 1,  # Reduced from 3 to allow smaller splits
                'random_state': 42,
                'early_stopping_rounds': 50
            }
        self.params = params
        self.model = None
        self.feature_importance = None
        self.scaler = StandardScaler()

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the XGBoost model"""
        print("\n" + "=" * 60)
        print("Training XGBoost Model")
        print("=" * 60)

        # Scale features
        print("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)

        # Extract early stopping rounds if present
        early_stopping = self.params.pop('early_stopping_rounds', None)

        self.model = xgboost.XGBRegressor(**self.params)

        if X_val is not None and y_val is not None and early_stopping is not None:
            X_val_scaled = self.scaler.transform(X_val)
            X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)

            eval_set = [(X_train_scaled, y_train), (X_val_scaled, y_val)]
            self.model.fit(
                X_train_scaled, y_train,
                eval_set=eval_set,
                verbose=False
            )
        else:
            self.model.fit(X_train_scaled, y_train)

        # Store feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        print(f"✓ Model trained successfully")
        print(f"✓ Number of trees: {self.model.n_estimators}")
        print(f"✓ Best iteration: {self.model.best_iteration if hasattr(self.model, 'best_iteration') else 'N/A'}")
        print(f"✓ Max depth: {self.params['max_depth']}")

    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet")

        # Scale features before prediction
        X_scaled = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

        return self.model.predict(X_scaled)

    def evaluate(self, X, y, dataset_name="Test"):
        """Evaluate model performance"""
        predictions = self.predict(X)

        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, predictions)
        r2 = r2_score(y, predictions)

        print(f"\n{dataset_name} Set Performance:")
        print(f"  MSE:  {mse:.8f}")
        print(f"  RMSE: {rmse:.8f}")
        print(f"  MAE:  {mae:.8f}")
        print(f"  R²:   {r2:.8f}")

        return {
            'predictions': predictions,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }

    def save_model(self, symbol, save_path='models'):
        """Save the trained model and scaler"""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'params': self.params,
            'feature_importance': self.feature_importance,
            'trained_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        filename = f"{save_path}/{symbol}_xgboost_model.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"\n✓ Model saved to: {filename}")
        return filename

    @classmethod
    def load_model(cls, symbol, load_path='models'):
        """Load a trained model and scaler"""
        filename = f"{load_path}/{symbol}_xgboost_model.pkl"

        if not os.path.exists(filename):
            raise FileNotFoundError(f"Model file not found: {filename}")

        with open(filename, 'rb') as f:
            model_data = pickle.load(f)

        # Create instance and restore model
        instance = cls(params=model_data['params'])
        instance.model = model_data['model']
        instance.scaler = model_data['scaler']
        instance.feature_importance = model_data['feature_importance']

        print(f"\n✓ Model loaded from: {filename}")
        print(f"  Trained on: {model_data['trained_date']}")

        return instance
