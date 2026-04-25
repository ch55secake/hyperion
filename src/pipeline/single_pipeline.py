from typing import Any

import numpy as np
import pandas as pd

from src.model import LightGBMStockPredictor
from src.model import XGBoostStockPredictor
from src.model.catboost.catboost_predictor import CatBoostStockPredictor
from src.optimise import StockModelOptimizer
from src.pipeline.base_pipeline import BaseTrainingPipeline
from src.simulation import TradingSimulator
from src.simulation.strategy.strategy_registry import StrategyRegistry
from src.util import logger
from src.writer import save_trained_model


class SingleModelTrainingPipeline(BaseTrainingPipeline):
    def __init__(self, model_type: str = "xgboost", *args, **kwargs):
        super().__init__(*args, **kwargs)
        if model_type not in ["xgboost", "lightgbm", "catboost"]:
            raise ValueError("model_type must be 'xgboost', 'lightgbm' or 'catboost'")
        self.model_type = model_type
        self._model_params = None

    def _create_model(self) -> Any:
        """
        Create either xgboost or lightgbm model
        :return:
        """
        match self.model_type:
            case "xgboost":
                return XGBoostStockPredictor(params=self._model_params)
            case "lightgbm":
                return LightGBMStockPredictor(params=self._model_params)
            case "catboost":
                return CatBoostStockPredictor(params=self._model_params)

    def _optimize_hyperparameters(self) -> Any:
        """

        :return:
        """
        logger.info(f"Running {self.model_type} hyperparameter optimization...")

        x_train_daily = self._test_train_data["train"]["daily"]
        y_train = self._test_train_data["train"]["targets"]
        x_test_daily = self._test_train_data["test"]["daily"]
        y_test = self._test_train_data["test"]["targets"]

        optimizer = StockModelOptimizer(x_train_daily, y_train, x_test_daily, y_test, n_trials=self.n_trials, n_jobs=1)

        if self.model_type == "xgboost":
            optimizer.optimize_xgboost()
            self._model_params = optimizer.best_xgb_params
        elif self.model_type == "lightgbm":
            optimizer.optimize_lightgbm()
            self._model_params = optimizer.best_lgb_params
        else:
            logger.warning(
                "Hyperparameter optimisation is not supported for model_type='%s'. "
                "Training will proceed with default parameters.",
                self.model_type,
            )

        optimizer.save_results(f"params/{self.model_type.upper()}_best_params.json")

    def _get_predictions(self):
        """
        Provide a raw dataframe for the model to make predictions on
        :return:
        """
        return self._model.predict(self._x_test_dict["daily"])

    def train(self) -> Any:
        """
        Train a single model on the training data which will be sampled at 1 day at a time
        :return:
        """
        if self._test_train_data is None:
            raise RuntimeError("Please run prepare_features() before train()")

        x_train_daily = self._test_train_data["train"]["daily"]
        y_train = self._test_train_data["train"]["targets"]
        x_test_daily = self._test_train_data["test"]["daily"]
        self._populate_test_train_data()

        logger.info("=" * 60)
        logger.info(f"Training Single {self.model_type.upper()} Model")
        logger.info("=" * 60)

        if self.should_optimise:
            self._optimize_hyperparameters()

        self._model = self._create_model()
        self._model.train(x_train_daily, y_train, x_test_daily, self._y_test)

        self._x_test_dict = {"daily": x_test_daily}
        test_results = self._model.evaluate(x_test_daily, self._y_test)
        self._test_results = test_results

        save_trained_model(
            self._model,
            f"{self.model_type.upper()}_MODEL",
            test_results,
            self.r2_save_threshold,
            self.r2_invalid_threshold,
        )
        logger.info(f"{self.model_type.upper()} model training complete!")

        return self

    def simulate(
        self,
        initial_capital: float = 10000,
        transaction_cost: float = 0.001,
        tickers: list[str] | None = None,
        strategy_name: str | None = None,
    ):
        """
        Simulate trading day by day using the trained single model's predictions.
        :param initial_capital: Starting capital for each ticker simulation.
        :param transaction_cost: Fractional transaction cost applied on each trade.
        :param tickers: Optional list of tickers to restrict simulation to.
        :param strategy_name: Name of a specific strategy to run; runs all registered
            strategies when ``None``.
        :return: pipeline instance
        """
        if self._test_results is None:
            raise RuntimeError("Predictions missing. Run train() first.")
        predictions = self._test_results.get("predictions")
        if predictions is None:
            raise RuntimeError("Predictions missing. Run train() first.")

        test_df = pd.DataFrame(
            {
                "symbol": self._symbols_test,
                "date": self._dates_test,
                "price": self._prices_test,
                "prediction": predictions,
                "actual_return": self._y_test,
            }
        )

        if tickers is not None:
            test_df = test_df[test_df["symbol"].isin(tickers)]
            logger.info(f"Filtering to {len(tickers)} tickers: {tickers}")

        unique_symbols = test_df["symbol"].unique()
        logger.info(f"Simulating {len(unique_symbols)} tickers")

        available_strategies = StrategyRegistry.list()
        if strategy_name is not None:
            if strategy_name not in available_strategies:
                raise ValueError(f"Strategy '{strategy_name}' not found. Available: {available_strategies}")
            strategies_to_run = [strategy_name]
        else:
            strategies_to_run = available_strategies

        logger.info(f"Running strategies: {strategies_to_run}")

        all_results = {}

        for strategy_key in strategies_to_run:
            logger.info(f"Strategy: {strategy_key}")
            strategy_results = {}

            for symbol in unique_symbols:
                try:
                    ticker_data = test_df[test_df["symbol"] == symbol].sort_values("date")
                    strategy_class = StrategyRegistry.get(strategy_key)

                    if len(ticker_data) < strategy_class.get_minimum_data_points():
                        logger.info(f"Skipping {symbol}: insufficient data ({len(ticker_data)} rows)")
                        continue

                    logger.info(f"{symbol} ({len(ticker_data)} samples)")
                    additional_data = strategy_class.get_extra_params(ticker_data.set_index("date")["price"])

                    simulator = TradingSimulator(initial_capital=initial_capital, transaction_cost=transaction_cost)
                    strategy = StrategyRegistry.create(
                        name=strategy_key, simulator=simulator, capital=initial_capital, **additional_data
                    )

                    ticker_data_reset = ticker_data.reset_index(drop=True)
                    threshold = np.percentile(np.abs(predictions), 75)

                    results = simulator.simulate(
                        predictions=ticker_data_reset["prediction"],
                        actual_returns=ticker_data_reset["actual_return"],
                        prices=ticker_data_reset["price"],
                        dates=ticker_data_reset["date"],
                        strategy=strategy,
                        threshold=threshold,
                    )
                    strategy_results[symbol] = results
                    logger.info(f"Final Value: ${results['final_value']:,.2f}")
                    logger.info(f"Return: {results['total_return'] * 100:.2f}%")
                except Exception as e:
                    logger.error("Error running %s on %s: %s", strategy_key, symbol, e, exc_info=True)

            all_results[strategy_key] = strategy_results

        return self
