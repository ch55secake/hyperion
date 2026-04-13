import pickle
import time
import traceback

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

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
        print(f"  Daily data shape: {x_train_daily.shape}")
        print(f"  Hourly data shape: {x_train_hourly.shape}")
        print(f"  Target shape: {y_train.shape}")
        print(f"  Daily index type: {type(x_train_daily.index)}")
        print(f"  Daily index first/last: {x_train_daily.index[0]} to {x_train_daily.index[-1]}")

        if not isinstance(x_train_daily.index, pd.DatetimeIndex):
            print("  Warning: Daily index is not DatetimeIndex, converting (fast method)...")
            x_train_daily.index = pd.to_datetime(x_train_daily.index, utc=True)

        if not isinstance(x_train_hourly.index, pd.DatetimeIndex):
            print("  Warning: Hourly index is not DatetimeIndex, converting (fast method)...")
            x_train_hourly.index = pd.to_datetime(x_train_hourly.index, utc=True)

        if not isinstance(y_train.index, pd.DatetimeIndex):
            print("  Warning: Target index is not DatetimeIndex, converting (fast method)...")
            y_train.index = pd.to_datetime(y_train.index, utc=True)

        print("  Sorting indices for optimal performance...")
        if not x_train_daily.index.is_monotonic_increasing:
            print("    Sorting daily index...")
            x_train_daily = x_train_daily.sort_index()
        if not x_train_hourly.index.is_monotonic_increasing:
            print("    Sorting hourly index...")
            x_train_hourly = x_train_hourly.sort_index()
        if not y_train.index.is_monotonic_increasing:
            print("    Sorting target index...")
            y_train = y_train.sort_index()

        self._meta_index = x_train_daily.index
        print(f"  Meta index type after conversion: {type(self._meta_index)}")
        print(f"  Meta index is sorted: {self._meta_index.is_monotonic_increasing}")

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

        print(f"  Created {len(base_models)} base model configs")
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

        print("\n" + "=" * 60)
        print("Training TimeSeriesStacker Model")
        print("=" * 60)
        print(f"Training samples: {len(x_train_daily)}")
        print(f"Testing samples: {len(x_test_daily)}")
        print(f"Unique stocks in test set: {self._symbols_test.nunique()}")

        if self.should_optimise:
            print("Running hyperparameter optimisation, this will take a while...")
            self._optimize_hyperparameters(x_train_daily, y_train, x_test_daily, self._y_test)

        print("\nCreating base models configuration...")

        print("  Aligning target index with daily features...")
        if not y_train.index.equals(x_train_daily.index):
            print("  WARNING: Target index doesn't match daily index, reindexing...")
            y_train = y_train.reindex(x_train_daily.index)

        base_models_config = self._create_base_models_config(x_train_daily, x_train_hourly, y_train)

        print(f"  Base models: {[bm['name'] for bm in base_models_config]}")
        print(f"  Meta index length: {len(self._meta_index)}")
        print(f"  Target length: {len(y_train)}")

        print("\nInitializing TimeSeriesStacker...")
        print("  This may take a moment for large datasets...")

        try:
            start_time = time.time()
            print("  Creating stacker object...")
            self._stacker = TimeSeriesStacker(
                base_models=base_models_config,
                meta_index=self._meta_index,
                target=y_train,
                n_splits=5,
                meta_model=Ridge(alpha=1.0),
            )
            elapsed = time.time() - start_time
            print(f"✓ TimeSeriesStacker initialized successfully in {elapsed:.2f}s")
        except Exception as e:
            print(f"✗ Error initializing TimeSeriesStacker: {e}")
            traceback.print_exc()
            raise

        print("\nGenerating out-of-fold predictions and training meta-learner...")
        oof_results = self._stacker.fit_meta()

        print("\nMeta-model OOF Performance:")
        print(f"  R² Score: {oof_results['meta_oof_df'].shape}")

        print("\nRetraining base models on full data for test predictions...")

        if not isinstance(x_test_daily.index, pd.DatetimeIndex):
            print("  Converting test daily index...")
            x_test_daily.index = pd.to_datetime(x_test_daily.index, utc=True)
        if not isinstance(x_test_hourly.index, pd.DatetimeIndex):
            print("  Converting test hourly index...")
            x_test_hourly.index = pd.to_datetime(x_test_hourly.index, utc=True)
        if not isinstance(self._y_test.index, pd.DatetimeIndex):
            print("  Converting test target index...")
            self._y_test.index = pd.to_datetime(self._y_test.index, utc=True)

        test_meta_index = x_test_daily.index

        print("  Concatenating train and test data...")
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

        print("  Updating stacker configuration...")
        self._stacker.base_models = base_models_full
        self._stacker.target = pd.concat([y_train, self._y_test])

        print("  Running fit_full_and_predict...")
        test_results = self._stacker.fit_full_and_predict(test_meta_index)

        self._meta_predictions = test_results["meta_preds"]
        self._base_predictions = test_results["base_preds"]

        self._test_results = {
            "predictions": self._meta_predictions.values,
            "base_predictions": self._base_predictions,
            "meta_oof": oof_results,
            "evals": test_results["evals"],
        }

        print("\nTest Set Performance:")
        if test_results["evals"]:
            for metric, value in test_results["evals"].items():
                print(f"  {metric.upper()}: {value:.4f}")

        model_name = "ALL_STOCKS"
        save_trained_model(self._stacker.meta_model, model_name, self._test_results)

        return self

    def _save_stacker(self, model_name: str):
        """Save the stacker model"""
        with open(f"models/stacker_{model_name}.pkl", "wb") as f:
            pickle.dump(self._stacker, f)
        print(f"Stacker saved to models/stacker_{model_name}.pkl")

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
            print(f"\n{'=' * 60}")
            print(f"Strategy: {strategy_key}")
            print(f"{'=' * 60}")

            strategy_results = {}

            for symbol in unique_symbols:
                try:
                    ticker_data = test_df[test_df["symbol"] == symbol].sort_values("date")
                    strategy_class = StrategyRegistry.get(strategy_key)

                    if len(ticker_data) < strategy_class.get_minimum_data_points():
                        print(f" Skipping {symbol}: insufficient data ({len(ticker_data)} rows)")
                        continue

                    print(f"\n--- {symbol} ({len(ticker_data)} samples) ---")

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

                    print(f"Final Value: ${results['final_value']:,.2f}")
                    print(f"Return: {results['total_return'] * 100:.2f}%")

                except Exception as e:
                    print(f" Error running {strategy_key} on {symbol}: {e}")
                    traceback.print_exc()

            all_results[strategy_key] = strategy_results

            print(f"\n{'=' * 60}")
            print(f"Summary for {strategy_key}")
            print(f"{'=' * 60}")

            if strategy_results:
                total_final_value = sum(r["final_value"] for r in strategy_results.values())
                avg_return = np.mean([r["total_return"] for r in strategy_results.values()])
                winning_tickers = sum(1 for r in strategy_results.values() if r["total_return"] > 0)

                print(f"Tickers simulated: {len(strategy_results)}")
                print(f"Total final value: ${total_final_value:,.2f}")
                print(f"Average return: {avg_return * 100:.2f}%")
                win_pct = winning_tickers / len(strategy_results) * 100
                print(
                    f"Winning tickers: {winning_tickers}/{len(strategy_results)} ({win_pct:.1f}%)"
                )

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
            print("Stacker not trained yet.")
            return None

        print("\n" + "=" * 60)
        print("Stacker Analysis")
        print("=" * 60)

        if hasattr(self._stacker.fitted_meta, "coef_"):
            coeffs = self._stacker.fitted_meta.coef_
            feature_names = self._stacker.meta_features.columns

            print("\nMeta-Model Weights:")
            for name, coef in zip(feature_names, coeffs):
                print(f"  {name}: {coef:.4f}")

        if self._base_predictions is not None:
            print("\nBase Model Prediction Correlations:")
            corr_matrix = self._base_predictions.corr()
            print(corr_matrix)

        return self
