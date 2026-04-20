from catboost import CatBoostRegressor, Pool

from src.model.model import Model
from src.util import get_device


class CatBoostStockPredictor(Model):
    """
    CatBoost model for stock price prediction
    """

    def __init__(self, params=None):
        if params is None:
            params = {
                # Core hyperparameters
                "iterations": 1500,
                "learning_rate": 0.05,
                "depth": 6,
                # Regularization
                "l2_leaf_reg": 5,
                "random_strength": 1,
                "bagging_temperature": 0.5,
                # Loss and evaluation
                "loss_function": "RMSE",
                "eval_metric": "RMSE",
                # Performance and stability
                "bootstrap_type": "Bayesian",
                "sampling_frequency": "PerTree",
                # GPU / CPU selection
                "task_type": "GPU" if get_device() == "cuda" else "CPU",
                # Early stopping and seeds
                "random_seed": 42,
                "early_stopping_rounds": 100,
                # Output
                "verbose": 100,
            }
        super().__init__("catboost", params)
        self.early_stopping_rounds = 50
        self.cat_features = ["ticker", "sector", "industry"]

        self.model = CatBoostRegressor(**params)

    def train(self, x_train, y_train, x_val=None, y_val=None):
        """
        Train a catboost model
        :param x_train:
        :param y_train:
        :param x_val:
        :param y_val:
        :return:
        """
        if self.cat_features:
            train_pool = Pool(data=x_train, label=y_train, cat_features=self.cat_features)
        else:
            train_pool = (x_train, y_train)

        eval_set = None
        if x_val is not None and y_val is not None:
            if self.cat_features:
                eval_set = Pool(data=x_val, label=y_val, cat_features=self.cat_features)
            else:
                eval_set = (x_val, y_val)

        self.model.fit(  # ty: ignore[unresolved-attribute]
            train_pool if self.cat_features else x_train,
            None if self.cat_features else y_train,
            eval_set=eval_set,
            early_stopping_rounds=self.early_stopping_rounds if eval_set else None,
            verbose=self.model.get_params()["verbose"],  # ty: ignore[unresolved-attribute]
        )

        return self

    def predict(self, x):
        """
        Make predictions with the catboost model
        :param x:
        :return:
        """
        return self.model.predict(self._prepare_prediction(x))  # ty: ignore[unresolved-attribute]
