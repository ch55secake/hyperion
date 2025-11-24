from src.lgb import LightGBMStockPredictor
from src.console import ConsoleFormatter
from src.optimise import StockModelOptimizer
from src.pipeline.base_pipeline import BaseTrainingPipeline
from src.stacker import StackedStockPredictor
from src.writer import save_trained_model
from src.xbg import XGBoostStockPredictor


class StackedModelTrainingPipeline(BaseTrainingPipeline):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._xgb_params = None
        self._lgb_params = None

    def _create_model(self):
        """
        Create a stacked model from both XGBoost and LightGBM models
        :return:
        """
        return StackedStockPredictor(
            {
                "daily": XGBoostStockPredictor(params=self._xgb_params),
                "hourly": LightGBMStockPredictor(params=self._lgb_params),
            }
        )

    def _get_predictions(self):
        """

        :return:
        """
        return self._model.predict(self._x_test_dict)

    def train(self):
        """
        Train both the daily and hourly models on the combined training data and then flatten them
        :return:
        """
        if self._test_train_data is None:
            raise Exception("Please run prepare_features(), before trying to run train()")

        x_train_daily = self._test_train_data["train"]["daily"]
        x_train_hourly = self._test_train_data["train"]["hourly"]
        y_train = self._test_train_data["train"]["targets"]

        x_test_daily = self._test_train_data["test"]["daily"]
        x_test_hourly = self._test_train_data["test"]["hourly"]
        self._y_test = self._test_train_data["test"]["targets"]
        self._dates_test = self._test_train_data["test"]["dates"]
        self._prices_test = self._test_train_data["test"]["prices"]
        self._symbols_test = self._test_train_data["test"]["symbols"]

        ConsoleFormatter.new_section("Training Single Model", new_lines_before_message=1)

        ConsoleFormatter.info(f"Training samples: {len(x_train_daily)}")
        ConsoleFormatter.info(f"Testing samples: {len(x_test_daily)}")
        ConsoleFormatter.info(f"Unique stocks in test set: {self._symbols_test.nunique()}")

        if self.should_optimise:
            ConsoleFormatter.info("Running hyperparameter optimisation, this will take a while...")
            self._optimize_hyperparameters(x_train_daily, y_train, x_test_daily, self._y_test)

        self._model = StackedStockPredictor(
            {
                "daily": XGBoostStockPredictor(params=self._xgb_params),
                "hourly": LightGBMStockPredictor(params=self._lgb_params),
            }
        )

        train_data = {
            "daily": (x_train_daily, y_train, x_test_daily, self._y_test),
            "hourly": (x_train_hourly, y_train, x_test_hourly, self._y_test),
        }

        self._model.train(train_data)

        self._x_test_dict = {"daily": x_test_daily, "hourly": x_test_hourly}
        test_results = self._model.evaluate(self._x_test_dict, self._y_test)

        model_name: str = "ALL_STOCKS"
        save_trained_model(self._model, model_name, test_results)

        return self

    def _optimize_hyperparameters(self, x_train_daily=None, y_train=None, x_test_daily=None, y_test=None):
        """
        Run optimization will only be run if the flag is enabled when the pipeline is instantiated.
        :param x_train_daily:
        :param y_train:
        :param x_test_daily:
        :param y_test:
        :return:
        """
        optimizer = StockModelOptimizer(x_train_daily, y_train, x_test_daily, y_test, n_trials=200, n_jobs=1)
        optimizer.optimize_both()
        optimizer.visualize_studies(save_path="plots/optuna")
        optimizer.save_results(f"params/ALL_STOCKS_best_params.json")

        self._xgb_params, self._lgb_params = optimizer.best_xgb_params, optimizer.best_lgb_params

        return self
