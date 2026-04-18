import logging
import pickle
import time
import traceback

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

logger = logging.getLogger(__name__)

from src.model import CatBoostStockPredictor
from src.model import LightGBMStockPredictor
from src.model import TimeSeriesStacker
from src.model import XGBoostStockPredictor
from src.optimise import StockModelOptimizer
from src.pipeline.base_pipeline import BaseTrainingPipeline
from src.simulation import TradingSimulator
from src.simulation.strategy.strategy_registry import StrategyRegistry
from src.writer import save_trained_model

# Required for the usage of the strategy registry
import src.simulation.strategy  # pylint: disable=unused-import


class TimeSeriesStackedModelTrainingPipeline(BaseTrainingPipeline):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._xgb_params = None
        self._lgb_params = None
        self._cat_params = None
        self._stacker = None
        self._meta_index = None
        self._test_results = None

        self._meta_predictions = None
        self._base_predictions = None

    def load_model(self):
        """
        Load a previously trained stacker model
        """
        with open("models/stacker_ALL_STOCKS.pkl", "rb") as f:
            self._stacker = pickle.load(f)
        return self

    def _create_model(self):
        """
        Create the TimeSeriesStacker model - implemented in train() method
        This is called internally during training
        """
        if self._stacker is None:
            raise Exception("Stacker not yet initialized. Call train() first.")
        return self._stacker

    def _create_base_models_config(self, x_train_daily, x_train_hourly, y_train):
        """
        Create configuration for base models with different feature sets/frequencies
        to ensure model diversity
        """
        logger.info("Daily data shape: %s", x_train_daily.shape)
        logger.info("Hourly data shape: %s", x_train_hourly.shape)
        logger.info("Target shape: %s", y_train.shape)
        logger.info("Daily index type: %s", type(x_train_daily.index))
        logger.info("Daily index first/last: %s to %s", x_train_daily.index[0], x_train_daily.index[-1])

        if not isinstance(x_train_daily.index, pd.DatetimeIndex):
            logger.warning("Daily index is not DatetimeIndex, converting (fast method)...")
            x_train_daily.index = pd.to_datetime(x_train_daily.index, utc=True)

        if not isinstance(x_train_hourly.index, pd.DatetimeIndex):
            logger.warning("Hourly index is not DatetimeIndex, converting (fast method)...")
            x_train_hourly.index = pd.to_datetime(x_train_hourly.index, utc=True)

        if not isinstance(y_train.index, pd.DatetimeIndex):
            logger.warning("Target index is not DatetimeIndex, converting (fast method)...")
            y_train.index = pd.to_datetime(y_train.index, utc=True)

        logger.info("Sorting indices for optimal performance...")
        if not x_train_daily.index.is_monotonic_increasing:
            logger.info("Sorting daily index...")
            x_train_daily = x_train_daily.sort_index()
        if not x_train_hourly.index.is_monotonic_increasing:
            logger.info("Sorting hourly index...")
            x_train_hourly = x_train_hourly.sort_index()
        if not y_train.index.is_monotonic_increasing:
            logger.info("Sorting target index...")
            y_train = y_train.sort_index()

        self._meta_index = x_train_daily.index
        logger.info("Meta index type after conversion: %s", type(self._meta_index))
        logger.info("Meta index is sorted: %s", self._meta_index.is_monotonic_increasing)

        base_models = [
            {
                "name": "xgb_daily",
                "model_factory": lambda: XGBoostStockPredictor(params=self._xgb_params),
                "X": x_train_daily,
                "y": y_train,
                "align": "mean",
            },
            {
                "name": "lgb_hourly",
                "model_factory": lambda: LightGBMStockPredictor(params=self._lgb_params),
                "X": x_train_hourly,
                "y": None,
                "align": "mean",
            },
            {
                "name": "cat_daily",
                "model_factory": lambda: CatBoostStockPredictor(params=self._cat_params),
                "X": x_train_daily,
                "y": y_train,
                "align": "mean",
            },
        ]

        logger.info("Created %d base model configs", len(base_models))
        return base_models

    def _get_predictions(self):
        """
        Get predictions from the stacker
        """
        if self._stacker is None:
            raise Exception("Stacker not trained. Run train() first.")

        return self._meta_predictions

    def train(self):
        """
        Train the TimeSeriesStacker with multiple base models
        """
        if self._test_train_data is None:
            raise Exception("Please run prepare_features(), before trying to run train()")

        x_train_daily = self._test_train_data["train"]["daily"]
        x_train_hourly = self._test_train_data["train"]["hourly"]
        y_train = self._test_train_data["train"]["targets"]

        x_test_daily = self._test_train_data["test"]["daily"]
        x_test_hourly = self._test_train_data["test"]["hourly"]
        self._populate_test_train_data()

        logger.info("Training TimeSeriesStacker Model")
        logger.info("Training samples: %d", len(x_train_daily))
        logger.info("Testing samples: %d", len(x_test_daily))
        logger.info("Unique stocks in test set: %d", self._symbols_test.nunique())

        if self.should_optimise:
            logger.info("Running hyperparameter optimisation, this will take a while...")
            self._optimize_hyperparameters(x_train_daily, y_train, x_test_daily, self._y_test)

        logger.info("Creating base models configuration...")

        logger.info("Aligning target index with daily features...")
        if not y_train.index.equals(x_train_daily.index):
            logger.warning("Target index doesn't match daily index, reindexing...")
            y_train = y_train.reindex(x_train_daily.index)

        base_models_config = self._create_base_models_config(x_train_daily, x_train_hourly, y_train)

        logger.info("Base models: %s", [bm['name'] for bm in base_models_config])
        logger.info("Meta index length: %d", len(self._meta_index))
        logger.info("Target length: %d", len(y_train))

        logger.info("Initializing TimeSeriesStacker...")

        try:
            start_time = time.time()
            logger.info("Creating stacker object...")
            self._stacker = TimeSeriesStacker(
                base_models=base_models_config,
                meta_index=self._meta_index,
                target=y_train,
                n_splits=5,
                meta_model=Ridge(alpha=1.0),
            )
            elapsed = time.time() - start_time
            logger.info("TimeSeriesStacker initialized successfully in %.2fs", elapsed)
        except Exception as e:
            logger.error("Error initializing TimeSeriesStacker: %s", e)
            traceback.print_exc()
            raise

        logger.info("Generating out-of-fold predictions and training meta-learner...")
        oof_results = self._stacker.fit_meta()

        logger.info("Meta-model OOF DataFrame shape: %s", oof_results['meta_oof_df'].shape)

        logger.info("Retraining base models on full data for test predictions...")

        if not isinstance(x_test_daily.index, pd.DatetimeIndex):
            logger.info("Converting test daily index...")
            x_test_daily.index = pd.to_datetime(x_test_daily.index, utc=True)
        if not isinstance(x_test_hourly.index, pd.DatetimeIndex):
            logger.info("Converting test hourly index...")
            x_test_hourly.index = pd.to_datetime(x_test_hourly.index, utc=True)
        if not isinstance(self._y_test.index, pd.DatetimeIndex):
            logger.info("Converting test target index...")
            self._y_test.index = pd.to_datetime(self._y_test.index, utc=True)

        test_meta_index = x_test_daily.index

        logger.info("Concatenating train and test data...")
        base_models_full = [
            {
                "name": "xgb_daily",
                "model_factory": lambda: XGBoostStockPredictor(params=self._xgb_params),
                "X": pd.concat([x_train_daily, x_test_daily]),
                "y": pd.concat([y_train, self._y_test]),
                "align": "mean",
            },
            {
                "name": "lgb_hourly",
                "model_factory": lambda: LightGBMStockPredictor(params=self._lgb_params),
                "X": pd.concat([x_train_hourly, x_test_hourly]),
                "y": None,
                "align": "mean",
            },
            {
                "name": "cat_daily",
                "model_factory": lambda: CatBoostStockPredictor(params=self._cat_params),
                "X": pd.concat([x_train_daily, x_test_daily]),
                "y": pd.concat([y_train, self._y_test]),
                "align": "mean",
            },
        ]

        logger.info("Updating stacker configuration...")
        self._stacker.base_models = base_models_full
        self._stacker.target = pd.concat([y_train, self._y_test])

        logger.info("Running fit_full_and_predict...")
        test_results = self._stacker.fit_full_and_predict(test_meta_index)

        self._meta_predictions = test_results["meta_preds"]
        self._base_predictions = test_results["base_preds"]

        self._test_results = {
            "predictions": self._meta_predictions.values,
            "base_predictions": self._base_predictions,
            "meta_oof": oof_results,
            "evals": test_results["evals"],
        }

        logger.info("Test Set Performance:")
        if test_results["evals"]:
            for metric, value in test_results["evals"].items():
                logger.info("  %s: %.4f", metric.upper(), value)

        model_name = "ALL_STOCKS"
        save_trained_model(self._stacker.meta_model, model_name, self._test_results)

        return self

    def _save_stacker(self, model_name: str):
        """Save the stacker model"""
        with open(f"models/stacker_{model_name}.pkl", "wb") as f:
            pickle.dump(self._stacker, f)
        logger.info("Stacker saved to models/stacker_%s.pkl", model_name)

    def simulate(self, initial_capital: float = 10000, tickers=None, strategy_name: str = None):
        """
        Use the stacked trained model to simulate trading day by day, per ticker
        """
        if tickers is None:
            tickers = ["AAPL"]

        predictions = self._test_results.get("predictions")
        if predictions is None:
            raise Exception("Predictions missing. Run train() first.")

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
            logger.info("Filtering to %d tickers: %s", len(tickers), tickers)

        unique_symbols = test_df["symbol"].unique()
        logger.info("Simulating %d tickers", len(unique_symbols))

        available_strategies = StrategyRegistry.list()
        if strategy_name is not None:
            if strategy_name not in available_strategies:
                raise ValueError(f"Strategy '{strategy_name}' not found. Available: {available_strategies}")
            strategies_to_run = [strategy_name]
        else:
            strategies_to_run = available_strategies

        logger.info("Running strategies: %s", strategies_to_run)

        all_results = {}

        for strategy_key in strategies_to_run:
            logger.info("Strategy: %s", strategy_key)

            strategy_results = {}

            for symbol in unique_symbols:
                try:
                    ticker_data = test_df[test_df["symbol"] == symbol].sort_values("date")
                    strategy_class = StrategyRegistry.get(strategy_key)

                    if len(ticker_data) < strategy_class.get_minimum_data_points():
                        logger.info("Skipping %s: insufficient data (%d rows)", symbol, len(ticker_data))
                        continue

                    logger.info("%s (%d samples)", symbol, len(ticker_data))

                    additional_data = strategy_class.get_extra_params(ticker_data.set_index("date")["price"])

                    simulator = TradingSimulator(initial_capital=initial_capital)
                    strategy = StrategyRegistry.create(
                        name=strategy_key, simulator=simulator, capital=initial_capital, **additional_data
                    )

                    ticker_data_reset = ticker_data.reset_index(drop=True)

                    train_predictions = predictions[self._split_idx :]

                    threshold = np.percentile(np.abs(train_predictions), 75)

                    results = simulator.simulate(
                        predictions=ticker_data_reset["prediction"],
                        actual_returns=ticker_data_reset["actual_return"],
                        prices=ticker_data_reset["price"],
                        dates=ticker_data_reset["date"],
                        strategy=strategy,
                        threshold=threshold,
                    )

                    strategy_results[symbol] = results

                    logger.info("Final Value: $%.2f", results['final_value'])
                    logger.info("Return: %.2f%%", results['total_return'] * 100)

                except Exception as e:
                    logger.error("Error running %s on %s: %s", strategy_key, symbol, e)
                    traceback.print_exc()

            all_results[strategy_key] = strategy_results

            logger.info("Summary for %s", strategy_key)

            if strategy_results:
                total_final_value = sum(r["final_value"] for r in strategy_results.values())
                avg_return = np.mean([r["total_return"] for r in strategy_results.values()])
                winning_tickers = sum(1 for r in strategy_results.values() if r["total_return"] > 0)

                logger.info("Tickers simulated: %d", len(strategy_results))
                logger.info("Total final value: $%.2f", total_final_value)
                logger.info("Average return: %.2f%%", avg_return * 100)
                win_pct = winning_tickers / len(strategy_results) * 100
                logger.info("Winning tickers: %d/%d (%.1f%%)", winning_tickers, len(strategy_results), win_pct)

        return self

    def _optimize_hyperparameters(self, x_train_daily=None, y_train=None, x_test_daily=None, y_test=None):
        """
        Run optimization for all base model types
        """
        optimizer = StockModelOptimizer(x_train_daily, y_train, x_test_daily, y_test, n_trials=200, n_jobs=1)
        optimizer.optimize_both()

        optimizer.visualize_studies(save_path="plots/optuna")
        optimizer.save_results("params/ALL_STOCKS_best_params.json")

        self._xgb_params, self._lgb_params = optimizer.best_xgb_params, optimizer.best_lgb_params

        return self

    def analyze_stacker_performance(self):
        """
        Analyze which base models contribute most to the ensemble
        """
        if self._stacker is None or self._stacker.fitted_meta is None:
            logger.warning("Stacker not trained yet.")
            return None

        logger.info("Stacker Analysis")

        if hasattr(self._stacker.fitted_meta, "coef_"):
            coeffs = self._stacker.fitted_meta.coef_
            feature_names = self._stacker.meta_features.columns

            logger.info("Meta-Model Weights:")
            for name, coef in zip(feature_names, coeffs):
                logger.info("  %s: %.4f", name, coef)

        if self._base_predictions is not None:
            logger.info("Base Model Prediction Correlations:\n%s", self._base_predictions.corr())

        return self
