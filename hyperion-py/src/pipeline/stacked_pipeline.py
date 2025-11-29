import numpy as np
import pandas as pd

from src.model import LightGBMStockPredictor
from src.model import XGBoostStockPredictor
from src.optimise import StockModelOptimizer
from src.pipeline.base_pipeline import BaseTrainingPipeline
from src.simulation import TradingSimulator
from src.simulation.strategy.strategy_registry import StrategyRegistry
from src.model import StackedStockPredictor
from src.writer import save_trained_model

from src.simulation.strategy import adaptive
from src.simulation.strategy import bb_reversion
from src.simulation.strategy import directional
from src.simulation.strategy import contrarian
from src.simulation.strategy import ema_cross
from src.simulation.strategy import hold_days
from src.simulation.strategy import hybrid_trend_ml
from src.simulation.strategy import momentum
from src.simulation.strategy import sltp
from src.simulation.strategy import sma_trend
from src.simulation.strategy import time_stop
from src.simulation.strategy import volatility_adjusted_threshold
from src.model import XGBoostStockPredictor
from src.data import StockDataDownloader


class StackedModelTrainingPipeline(BaseTrainingPipeline):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._xgb_params = None
        self._lgb_params = None

    def load_model(self):
        """
        Load a previously trained model instead of training a new one
        :return:
        """
        self._model = StackedStockPredictor.load_model("ALL_STOCKS")

        return self

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
        Grab the dictionary of the both daily and hourly models and pass it to the stacked model
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
        self._populate_test_train_data()

        print("\n" + "=" * 60)
        print("Training Single Model")
        print("=" * 60)
        print(f"Training samples: {len(x_train_daily)}")
        print(f"Testing samples: {len(x_test_daily)}")
        print(f"Unique stocks in test set: {self._symbols_test.nunique()}")

        if self.should_optimise:
            print("Running hyperparameter optimisation, this will take a while...")
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
        self._test_results = self._model.evaluate(self._x_test_dict, self._y_test)

        model_name: str = "ALL_STOCKS"
        save_trained_model(self._model, model_name, self._test_results)

        return self

    def simulate(self, initial_capital: float = 10000, tickers=None, strategy_name: str = None):
        """
        Use the stacked trained model to simulate trading day by day, per ticker
        :return:
        """
        if tickers is None:
            tickers = ["AAPL"]
        predictions = self._test_results.get("predictions")
        if predictions is None:
            print(" Predictions missing, computing via predictor.predict()")
            predictor = self._x_test_dict.get("predictor")
            predictions = predictor.predict(self._x_test_dict)

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
            print(f"Filtering to {len(tickers)} tickers: {tickers}")

        unique_symbols = test_df["symbol"].unique()
        print(f"\nSimulating {len(unique_symbols)} tickers")

        available_strategies = StrategyRegistry.list()
        if strategy_name is not None:
            if strategy_name not in available_strategies:
                raise ValueError(f"Strategy '{strategy_name}' not found. Available: {available_strategies}")
            strategies_to_run = [strategy_name]
        else:
            strategies_to_run = available_strategies

        print(f"Running strategies: {strategies_to_run}\n")

        all_results = {}

        for strategy_key in strategies_to_run:
            print(f"\n{'='*60}")
            print(f"Strategy: {strategy_key}")
            print(f"{'='*60}")

            strategy_results = {}

            for symbol in unique_symbols:
                try:
                    ticker_data = test_df[test_df["symbol"] == symbol].sort_values("date")
                    strategy_class = StrategyRegistry.get(strategy_key)

                    if len(ticker_data) < strategy_class.get_minimum_data_points():
                        print(f" Skipping {symbol}: insufficient data ({len(ticker_data)} rows)")
                        continue

                    print(f"\n--- {symbol} ({len(ticker_data)} trades) ---")

                    additional_data = strategy_class.get_extra_params(ticker_data.set_index("date")["price"])

                    simulator = TradingSimulator(initial_capital=initial_capital)
                    strategy = StrategyRegistry.create(
                        name=strategy_key, simulator=simulator, capital=initial_capital, **additional_data
                    )

                    ticker_data_reset = ticker_data.reset_index(drop=True)

                    results = simulator.simulate(
                        predictions=ticker_data_reset["prediction"],
                        actual_returns=ticker_data_reset["actual_return"],
                        prices=ticker_data_reset["price"],
                        dates=ticker_data_reset["date"],
                        strategy=strategy,
                    )

                    strategy_results[symbol] = results

                    print(f"Final Value: ${results['final_value']:,.2f}")
                    print(f"Return: {results['total_return']*100:.2f}%")

                except Exception as e:
                    print(f" Error running {strategy_key} on {symbol}: {e}")
                    import traceback

                    traceback.print_exc()

            all_results[strategy_key] = strategy_results

            print(f"\n{'='*60}")
            print(f"Summary for {strategy_key}")
            print(f"{'='*60}")

            if strategy_results:
                total_final_value = sum(r["final_value"] for r in strategy_results.values())
                avg_return = np.mean([r["total_return"] for r in strategy_results.values()])
                winning_tickers = sum(1 for r in strategy_results.values() if r["total_return"] > 0)

                print(f"Tickers simulated: {len(strategy_results)}")
                print(f"Total final value: ${total_final_value:,.2f}")
                print(f"Average return: {avg_return*100:.2f}%")
                print(
                    f"Winning tickers: {winning_tickers}/{len(strategy_results)} ({winning_tickers/len(strategy_results)*100:.1f}%)"
                )

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
