from typing import Any

from src.lgb import LightGBMStockPredictor
from src.console import ConsoleFormatter
from src.optimise import StockModelOptimizer
from src.pipeline.base_pipeline import BaseTrainingPipeline
from src.writer import save_trained_model
from src.xbg import XGBoostStockPredictor


class SingleModelTrainingPipeline(BaseTrainingPipeline):

    def __init__(self, model_type: str = "xgboost", *args, **kwargs):
        super().__init__(*args, **kwargs)
        if model_type not in ["xgboost", "lightgbm"]:
            raise ValueError("model_type must be 'xgboost' or 'lightgbm'")
        self.model_type = model_type
        self._model_params = None

    def _create_model(self) -> Any:
        """
        Create either xgboost or lightgbm model
        :return:
        """
        if self.model_type == "xgboost":
            return XGBoostStockPredictor(params=self._model_params)
        else:
            return LightGBMStockPredictor(params=self._model_params)

    def _optimize_hyperparameters(self) -> Any:
        """

        :return:
        """
        ConsoleFormatter.info(f"Running {self.model_type} hyperparameter optimization...")

        x_train_daily = self._test_train_data["train"]["daily"]
        y_train = self._test_train_data["train"]["targets"]
        x_test_daily = self._test_train_data["test"]["daily"]
        y_test = self._test_train_data["test"]["targets"]

        optimizer = StockModelOptimizer(x_train_daily, y_train, x_test_daily, y_test, n_trials=500, n_jobs=1)

        if self.model_type == "xgboost":
            optimizer.optimize_xgboost()
            self._model_params = optimizer.best_xgb_params
        else:
            optimizer.optimize_lightgbm()
            self._model_params = optimizer.best_lgb_params

        optimizer.save_results(f"params/{self.model_type.upper()}_best_params.json")

    def _get_predictions(self):
        """
        Provide a raw dataframe for the model to make predictions on
        :return:
        """
        return self._model.predict(self._x_test_dict["daily"])

    def train(self) -> Any:
        """

        :return:
        """
        if self._test_train_data is None:
            raise RuntimeError("Please run prepare_features() before train()")

        x_train_daily = self._test_train_data["train"]["daily"]
        y_train = self._test_train_data["train"]["targets"]
        x_test_daily = self._test_train_data["test"]["daily"]
        self._y_test = self._test_train_data["test"]["targets"]
        self._dates_test = self._test_train_data["test"]["dates"]
        self._prices_test = self._test_train_data["test"]["prices"]
        self._symbols_test = self._test_train_data["test"]["symbols"]

        ConsoleFormatter.new_section(f"Training Single {self.model_type.upper()} Model", new_lines_before_message=1)

        if self.should_optimise:
            self._optimize_hyperparameters()

        self._model = self._create_model()
        self._model.train(x_train_daily, y_train, x_test_daily, self._y_test)

        self._x_test_dict = {"daily": x_test_daily}
        test_results = self._model.evaluate(x_test_daily, self._y_test)

        save_trained_model(self._model, f"{self.model_type.upper()}_MODEL", test_results)
        ConsoleFormatter.success(f"{self.model_type.upper()} model training complete!", new_lines_before_message=1)

        return self
