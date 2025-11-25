import pandas as pd
import xgboost

from ..model import Model

class XGBoostStockPredictor(Model):
    """
    XGBoost model for stock price prediction with categorical feature support
    """

    def __init__(self, params=None):
        if params is None:
            params = {
                "objective": "reg:squarederror",
                "learning_rate": 0.04,
                "max_depth": 4,
                "min_child_weight": 1,
                "subsample": 0.7,
                "colsample_bytree": 0.7,
                "lambda": 0.6,
                "alpha": 0.3,
                "gamma": 0.0,
                "n_estimators": 500,
                "tree_method": "hist",
                "seed": 42,
                "enable_categorical": True,
            }
        super().__init__("xgboost", params=params)

    def train(self, x_train, y_train, x_val=None, y_val=None):
        """Train the XGBoost model"""
        print("\n" + "=" * 60)
        print("Training XGBoost Model")
        print("=" * 60)

        x_train_processed, object_cols = self._prepare_columns(x_train)

        print("\nDataFrame dtypes before fitting:")
        for col in x_train_processed.columns:
            if x_train_processed[col].dtype.name in ["object", "category"]:
                print(f"  {col}: {x_train_processed[col].dtype}")

        early_stopping = self.params.pop("early_stopping_rounds", None)

        enable_categorical = self.params.get("enable_categorical", False)
        if self.categorical_columns and not enable_categorical:
            print("⚠️ Warning: Categorical columns detected but enable_categorical not set. Enabling it.")
            self.params["enable_categorical"] = True

        self.model = xgboost.XGBRegressor(**self.params)

        if x_val is not None and y_val is not None and early_stopping is not None:
            x_val_processed = self._prepare_numeric_and_categorical_columns(x_val.copy(), object_cols, x_val)

            eval_set = [(x_train_processed, y_train), (x_val_processed, y_val)]
            self.model.fit(x_train_processed, y_train, eval_set=eval_set, verbose=False)
        else:
            self.model.fit(x_train_processed, y_train)

        feature_names = x_train_processed.columns.tolist()
        feature_importances = self.model.feature_importances_

        if len(feature_names) != len(feature_importances):
            print(
                f"⚠️ Warning: Feature name count ({len(feature_names)}) doesn't match importance count "
                f"({len(feature_importances)})"
            )
            feature_names = [f"feature_{i}" for i in range(len(feature_importances))]

        self.feature_importance = pd.DataFrame(
            {"feature": feature_names, "importance": feature_importances}
        ).sort_values("importance", ascending=False)

        print("✓ Model trained successfully")
        print(f"✓ Number of trees: {self.model.n_estimators}")
        print(f"✓ Best iteration: {self.model.best_iteration if hasattr(self.model, 'best_iteration') else 'N/A'}")
        print(f"✓ Max depth: {self.params['max_depth']}")

        print("\nTop 10 Most Important Features:")
        print(self.feature_importance.head(10).to_string(index=False))

    def predict(self, x):
        """Make predictions"""
        return self.model.predict(self._prepare_prediction(x))
