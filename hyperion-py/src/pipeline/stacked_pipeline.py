import traceback

import numpy as np
import pandas as pd
from typing_extensions import override

from src.data import StockDataDownloader
from src.feature import FeatureEngineering
from src.model import LightGBMStockPredictor
from src.model import StackedStockPredictor
from src.model import XGBoostStockPredictor
from src.optimise import StockModelOptimizer
from src.pipeline.base_pipeline import BaseTrainingPipeline
from src.simulation import TradingSimulator
from src.simulation.strategy.strategy_registry import StrategyRegistry
from src.writer import save_trained_model


# Required for the usage of the strategy registry
import src.simulation.strategy


class StackedModelTrainingPipeline(BaseTrainingPipeline):

    def __init__(self, intervals: list[str], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.intervals = intervals
        self._xgb_params = None
        self._lgb_params = None

        self.default_interval = self.intervals[0]

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
                "1d": XGBoostStockPredictor(params=self._xgb_params),
                "1h": LightGBMStockPredictor(params=self._lgb_params),
            }
        )

    @override
    def download_data(self):
        """
        Download both hourly and daily data instead of populating just daily data
        :return:
        """
        print("\n" + "=" * 60)
        print("Downloading and Preparing Data for All Stocks")
        print("=" * 60)

        if self.symbols is None:
            raise Exception("Please run read_tickers(), before trying to run download_data()")

        self._stock_data = {}
        for interval in self.intervals:
            self._downloader = StockDataDownloader(self.symbols, period=self.period, interval=interval)
            self._stock_data[interval] = self._downloader.download_data()

        if not self._stock_data:
            print("⚠️  No data downloaded. Exiting.")
            return None

        return self

    def __create_categorical_features(self, train, test):
        """
        fuck this
        :param train:
        :param test:
        :return:
        """
        categorical_cols: list[str] = ["ticker", "sector", "industry"]
        for interval in self.intervals:
            for col in categorical_cols:
                if col in train[interval].columns:
                    train[interval][col] = train[interval][col].astype("category")
                    test[interval][col] = test[interval][col].astype("category")
                    print(f" {interval} {col}: {train[interval][col].nunique()} unique values")

        return self, train, test

    @override
    def prepare_features(self):
        """
        Blalalalalalal
        :return:
        """

        if self._stock_data is None:
            raise Exception("Please run download_data(), before trying to run prepare_features()")

        train_features = {interval: [] for interval in self.intervals}
        train_targets = {interval: [] for interval in self.intervals}
        train_dates = {interval: [] for interval in self.intervals}
        train_prices = {interval: [] for interval in self.intervals}
        train_symbols = {interval: [] for interval in self.intervals}

        test_features = {interval: [] for interval in self.intervals}
        test_targets = {interval: [] for interval in self.intervals}
        test_dates = {interval: [] for interval in self.intervals}
        test_prices = {interval: [] for interval in self.intervals}
        test_symbols = {interval: [] for interval in self.intervals}

        for interval in self.intervals:
            for symbol in self.symbols:
                try:
                    print(f"\nProcessing {symbol}...")

                    features = FeatureEngineering(self._stock_data[interval][symbol])
                    features.create_target_features()
                    x, y, dates, prices, _ = features.prepare_features()

                    x = self._add_stock_features(x, symbol)

                    self._split_idx = int(len(x) * (1 - self.test_size))

                    train_features[interval].append(x.iloc[: self._split_idx])
                    train_targets[interval].append(y.iloc[: self._split_idx])
                    train_dates[interval].append(pd.Series(dates[: self._split_idx], index=dates[: self._split_idx]))
                    train_prices[interval].append(prices.iloc[: self._split_idx])
                    train_symbols[interval].extend([symbol] * self._split_idx)

                    test_features[interval].append(x.iloc[self._split_idx :])
                    test_targets[interval].append(y.iloc[self._split_idx :])
                    test_dates[interval].append(pd.Series(dates[self._split_idx :], index=dates[self._split_idx :]))
                    test_prices[interval].append(prices.iloc[self._split_idx :])
                    test_symbols[interval].extend([symbol] * (len(x) - self._split_idx))

                    print(f"  ✓ {symbol}: {self._split_idx} train samples, {len(x) - self._split_idx} test samples")

                except Exception as e:
                    print(f"  ✗ Error processing {symbol}: {str(e)}")
                    traceback.print_exc()
                    continue

        print("\n" + "=" * 60)
        print("Combining Training Data")
        print("=" * 60)
        train_intervals = dict()
        for interval in self.intervals:
            train_intervals[interval] = pd.concat(train_features[interval], axis=0, ignore_index=False)

        # Store targets per interval for proper y_test organization
        train_targets_dict = dict()
        for interval in self.intervals:
            train_targets_dict[interval] = pd.concat(train_targets[interval], axis=0, ignore_index=False)

        train_targets = train_targets_dict[self.default_interval]  # Keep for compatibility
        train_dates = pd.concat(train_dates[self.default_interval], axis=0, ignore_index=False)
        train_prices = pd.concat(train_prices[self.default_interval], axis=0, ignore_index=False)

        # Combine symbols from default interval only
        combined_train_symbols = train_symbols[self.default_interval]
        train_symbols_series = pd.Series(combined_train_symbols, index=train_intervals[self.default_interval].index)

        print("Combining Test Data")
        test_intervals = dict()
        for interval in self.intervals:
            test_intervals[interval] = pd.concat(test_features[interval], axis=0, ignore_index=False)

        # Store targets per interval for proper y_test organization
        test_targets_dict = dict()
        for interval in self.intervals:
            test_targets_dict[interval] = pd.concat(test_targets[interval], axis=0, ignore_index=False)

        test_targets = test_targets_dict[self.default_interval]  # Keep for compatibility
        test_dates = pd.concat(test_dates[self.default_interval], axis=0, ignore_index=False)
        test_prices = pd.concat(test_prices[self.default_interval], axis=0, ignore_index=False)

        test_symbols_series = pd.Series(
            test_symbols[self.default_interval], index=test_intervals[self.default_interval].index
        )

        print("\nConverting categorical columns...")
        _, train_intervals, test_intervals = self.__create_categorical_features(train_intervals, test_intervals)

        print(f"✓ Total train samples: {len(train_intervals[self.default_interval])}")
        print(f"✓ Total test samples: {len(test_intervals[self.default_interval])}")
        print(f"✓ Number of stocks: {len(self.symbols)}")
        print(f"✓ Stocks in test set: {test_symbols_series.nunique()}")
        print(f"✓ Features per timeframe: {len(train_intervals[self.default_interval].columns)}")

        self._test_train_data = {
            "train": {
                **{interval: train_intervals[interval] for interval in self.intervals},
                "targets": train_targets_dict,  # Now stores targets per interval
                "dates": train_dates,
                "prices": train_prices,
                "symbols": train_symbols_series,
            },
            "test": {
                **{interval: test_intervals[interval] for interval in self.intervals},
                "targets": test_targets_dict,  # Now stores targets per interval
                "dates": test_dates,
                "prices": test_prices,
                "symbols": test_symbols_series,
            },
        }

        return self

    def _populate_test_train_data(self):
        """
        Populate test train data with support for multiple intervals
        Override base method to handle interval-specific targets
        :return:
        """
        # Store targets per interval for proper multi-interval handling
        if isinstance(self._test_train_data["test"]["targets"], dict):
            self._y_test_dict = self._test_train_data["test"]["targets"]
            self._y_test = self._y_test_dict[self.default_interval]  # Keep for compatibility
        else:
            # Backward compatibility: targets is a single series (old format)
            self._y_test = self._test_train_data["test"]["targets"]
            self._y_test_dict = {interval: self._y_test for interval in self.intervals}

        self._dates_test = self._test_train_data["test"]["dates"]
        self._prices_test = self._test_train_data["test"]["prices"]
        self._symbols_test = self._test_train_data["test"]["symbols"]

    def _align_targets_across_intervals(self):
        """
        Align y_test targets across different time intervals to ensure consistency.
        Uses the default interval as the reference timeline.

        Returns:
            dict: Aligned targets per interval
        """
        aligned_targets = {}

        # Use default interval as the reference timeline
        reference_index = self._y_test_dict[self.default_interval].index

        for interval in self.intervals:
            if interval == self.default_interval:
                aligned_targets[interval] = self._y_test_dict[interval]
            else:
                # Align other intervals to reference using appropriate method
                aligned_targets[interval] = self._align_to_reference(
                    self._y_test_dict[interval], reference_index, method="ffill"
                )

        return aligned_targets

    def _align_to_reference(self, targets_series, reference_index, method="ffill"):
        """
        Align targets from one interval to reference index.

        Args:
            targets_series (pd.Series): Target series to align
            reference_index (pd.Index): Reference index to align to
            method (str): Alignment method ("ffill", "mean")

        Returns:
            pd.Series: Aligned targets series
        """
        if method == "ffill":
            # Simple forward fill alignment - truncate to minimum length
            min_length = min(len(targets_series), len(reference_index))

            # Use reference_index and truncate both to same length
            aligned_index = reference_index[:min_length]
            aligned_values = targets_series.iloc[:min_length].values

            return pd.Series(aligned_values, index=aligned_index)

        elif method == "mean":
            # Resample and take mean within each target period
            try:
                # Try to infer frequency from reference index
                freq = pd.infer_freq(reference_index)
                if freq is not None:
                    resampled = targets_series.resample(freq).mean()
                else:
                    resampled = targets_series.resample("D").mean()
            except Exception:
                resampled = targets_series.resample("D").mean()

            # Reindex to exact reference_index and forward-fill
            aligned = resampled.reindex(reference_index, method="ffill")
            return aligned

        else:
            raise ValueError(f"Unknown alignment method: {method}")

    def _validate_data_consistency(self):
        """
        Validate that all test data components are properly aligned.

        Returns:
            list: Validation results for each interval
        """
        validations = []

        for interval in self.intervals:
            x_test_len = len(self._test_train_data["test"][interval])
            y_test_len = len(self._y_test_dict[interval]) if interval in self._y_test_dict else 0

            validation = {
                "interval": interval,
                "x_test_length": x_test_len,
                "y_test_length": y_test_len,
                "lengths_match": x_test_len == y_test_len,
                "x_test_index": self._test_train_data["test"][interval].index[:5].tolist() if x_test_len > 0 else [],
                "y_test_index": self._y_test_dict[interval].index[:5].tolist() if y_test_len > 0 else [],
            }

            # Check index alignment if lengths match
            if validation["lengths_match"] and x_test_len > 0:
                x_index = self._test_train_data["test"][interval].index
                y_index = self._y_test_dict[interval].index
                validation["indices_equal"] = x_index.equals(y_index)

            validations.append(validation)

        # Log validation results
        print("\n" + "=" * 60)
        print("Data Consistency Validation")
        print("=" * 60)
        for val in validations:
            print(f"Interval: {val['interval']}")
            print(f"  X_test length: {val['x_test_length']}")
            print(f"  Y_test length: {val['y_test_length']}")
            print(f"  Lengths match: {val['lengths_match']}")
            if "indices_equal" in val:
                print(f"  Indices equal: {val['indices_equal']}")

        return validations

    def _ensure_prediction_alignment(self, predictions):
        """
        Ensure y_test aligns with prediction array indices.

        Args:
            predictions: Model predictions array

        Returns:
            pd.Series: Aligned y_test series
        """
        if len(predictions) != len(self._y_test):
            min_len = min(len(predictions), len(self._y_test))
            print(
                f"Warning: Prediction length ({len(predictions)}) and y_test length ({len(self._y_test)}) mismatch. Using {min_len} samples."
            )
            aligned_y_test = self._y_test.iloc[:min_len].reset_index(drop=True)
        else:
            aligned_y_test = self._y_test.reset_index(drop=True)

        return aligned_y_test

    def _get_predictions(self):
        """
        Grab the dictionary of the both daily and hourly models and pass it to the stacked model
        :return:
        """
        return self._model.predict(self._x_test_dict)

    def train(self):
        """
        Train both the daily and hourly models on the combined training data using interval-specific targets.
        Uses multi-interval target organization and alignment for proper model training.
        :return: self
        """
        if self._test_train_data is None:
            raise Exception("Please run prepare_features(), before trying to run train()")

        x_train = dict()
        x_test = dict()

        for interval in self.intervals:
            x_train[interval] = self._test_train_data["train"][interval]
            # TODO: this is debug print can be removed soon
            print(f"This is x_train[{interval}]: {x_train[interval]}")
            x_test[interval] = self._test_train_data["test"][interval]
            print(f"This is x_test[{interval}]: {x_test[interval]}")

        # Get training targets per interval
        if isinstance(self._test_train_data["train"]["targets"], dict):
            y_train_dict = self._test_train_data["train"]["targets"]
            y_train = y_train_dict[self.default_interval]  # For compatibility
        else:
            # Backward compatibility
            y_train_dict = {interval: self._test_train_data["train"]["targets"] for interval in self.intervals}
            y_train = self._test_train_data["train"]["targets"]

        self._populate_test_train_data()

        # Validate data consistency
        self._validate_data_consistency()

        print("\n" + "=" * 60)
        print("Training Stacked Model")
        print("=" * 60)
        print(f"Training samples: {len(x_train[self.default_interval])}")
        print(f"Testing samples: {len(x_test[self.default_interval])}")
        print(f"Unique stocks in test set: {self._symbols_test.nunique()}")

        if self.should_optimise:
            print("Running hyperparameter optimisation, this will take a while...")
            # self._optimize_hyperparameters(x_train_daily, y_train, x_test_daily, self._y_test)

        self._model = StackedStockPredictor(
            {
                "1d": XGBoostStockPredictor(params=self._xgb_params),
                "1h": LightGBMStockPredictor(params=self._lgb_params),
            }
        )

        # Use interval-specific targets for training data
        train_data = {
            interval: (x_train[interval], y_train_dict[interval], x_test[interval], self._y_test_dict[interval])
            for interval in self.intervals
        }

        self._model.train(train_data)

        self._x_test_dict = {interval: x_test[interval] for interval in self.intervals}

        # Align targets for evaluation
        aligned_targets = self._align_targets_across_intervals()
        self._test_results = self._model.evaluate(self._x_test_dict, aligned_targets[self.default_interval])

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
            predictions = self._get_predictions()

        # Ensure all data aligns with predictions
        min_len = min(
            len(predictions), len(self._y_test), len(self._symbols_test), len(self._dates_test), len(self._prices_test)
        )

        if len(predictions) != min_len:
            print(f"Warning: Truncating all test data to {min_len} samples for alignment")

        # Truncate all arrays to same length
        # Debug the actual lengths
        print(f"DEBUG: predictions length: {len(predictions)}")
        print(
            f"DEBUG: symbols_test type: {type(self._symbols_test)}, length: {len(self._symbols_test) if self._symbols_test is not None else 'None'}"
        )
        print(
            f"DEBUG: dates_test type: {type(self._dates_test)}, length: {len(self._dates_test) if self._dates_test is not None else 'None'}"
        )
        print(
            f"DEBUG: prices_test type: {type(self._prices_test)}, length: {len(self._prices_test) if self._prices_test is not None else 'None'}"
        )

        # Use predictions length as reference
        pred_len = len(predictions)

        # Ensure y_test aligns with predictions
        aligned_y_test = self._ensure_prediction_alignment(predictions)
        print(f"DEBUG: aligned_y_test length: {len(aligned_y_test) if aligned_y_test is not None else 'None'}")

        # Simple approach - create dict with matching lengths
        test_data = {
            "prediction": predictions[:pred_len],
        }

        # Add other arrays only if they exist and match
        if self._symbols_test is not None and len(self._symbols_test) >= pred_len:
            test_data["symbol"] = self._symbols_test.iloc[:pred_len].reset_index(drop=True)
        else:
            test_data["symbol"] = []

        if self._dates_test is not None and len(self._dates_test) >= pred_len:
            test_data["date"] = self._dates_test.iloc[:pred_len].reset_index(drop=True)
        else:
            test_data["date"] = []

        if self._prices_test is not None and len(self._prices_test) >= pred_len:
            test_data["price"] = self._prices_test.iloc[:pred_len].reset_index(drop=True)
        else:
            test_data["price"] = []

        if aligned_y_test is not None and len(aligned_y_test) >= pred_len:
            if hasattr(aligned_y_test, "iloc"):
                test_data["actual_return"] = aligned_y_test.iloc[:pred_len].reset_index(drop=True)
            else:
                test_data["actual_return"] = aligned_y_test[:pred_len]
        else:
            test_data["actual_return"] = []

        test_df = pd.DataFrame(test_data)

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

                    simulator = TradingSimulator(initial_capital=int(initial_capital))
                    strategy = StrategyRegistry.create(
                        name=strategy_key, simulator=simulator, capital=int(initial_capital), **additional_data
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
                        threshold=threshold if isinstance(threshold, str) else "auto",
                    )

                    strategy_results[symbol] = results

                    print(f"Final Value: ${results['final_value']:,.2f}")
                    print(f"Return: {results['total_return'] * 100:.2f}%")

                except Exception as e:
                    print(f" Error running {strategy_key} on {symbol}: {e}")
                    import traceback

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
                print(
                    f"Winning tickers: {winning_tickers}/{len(strategy_results)} ({winning_tickers / len(strategy_results) * 100:.1f}%)"
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
