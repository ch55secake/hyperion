import traceback
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

from src.data import StockDataDownloader
from src.feature import FeatureEngineering
from src.console import ConsoleFormatter


class BaseTrainingPipeline(ABC):
    """
    Base class for training pipelines
    """

    def __init__(
        self,
        symbols: list[str] = None,
        period: str = "2y",
        interval: str = "1d",
        test_size: float = 0.2,
        should_optimise: bool = False,
    ):
        self.symbols = symbols
        self.period = period
        self.interval = interval
        self.test_size = test_size
        self.should_optimise = should_optimise

        # Unfortunately, lots of state management but hey ho
        # State
        self._downloader = None
        self._stock_data = None
        self._model = None
        self._results = None
        self._test_train_data = None

        # References
        self._symbols_test = None
        self._x_test_dict = None
        self._y_test = None
        self._dates_test = None
        self._prices_test = None
        self._xgb_params = None
        self._lgb_params = None

    def read_tickers(self):
        """
        Reads tickers from the resources/tickers.txt file and stores them in the symbol attribute
        :return: pipeline instance
        """
        if self.symbols is None:
            symbols: list[str] = []
            with open("resources/tickers.txt", "r") as f:
                for line in f:
                    symbols.append(line.strip())

            self.symbols = symbols
            ConsoleFormatter.info(f"Read {len(symbols)} tickers from resources/tickers.txt")

        return self

    def download_data(self):
        """
        Download the historical stock data for all tickers provided in the resources/tickers.txt
        :return: pipeline instance but should also update the _stock_data attribute
        """
        ConsoleFormatter.new_section("Downloading and Preparing Data for All Stocks", new_lines_before_message=1)

        if self.symbols is None:
            raise Exception("Please run read_tickers(), before trying to run download_data()")

        self._downloader = StockDataDownloader(self.symbols, period=self.period, interval=self.interval)
        self._stock_data = self._downloader.download_data()

        if not self._stock_data:
            ConsoleFormatter.warning("No data downloaded. Exiting.")
            return None

        return self

    def prepare_features(self):
        """
        Prepare features for training and testing the model
        :return:
        """

        if self._stock_data is None:
            raise Exception("Please run download_data(), before trying to run prepare_features()")

        train_daily_features = []
        train_hourly_features = []
        train_targets = []
        train_dates = []
        train_prices = []
        train_symbols = []

        test_daily_features = []
        test_hourly_features = []
        test_targets = []
        test_dates = []
        test_prices = []
        test_symbols = []

        for symbol in self.symbols:
            try:
                ConsoleFormatter.info(f"Processing {symbol}...", new_lines_before_message=1)

                # Daily features
                features_daily = FeatureEngineering(self._stock_data[symbol])
                features_daily.create_target_features()
                x_daily, y_daily, dates_daily, prices_daily, _ = features_daily.prepare_features()

                # Add stock-specific features
                x_daily = self._add_stock_features(x_daily, symbol)

                # Hourly features
                features_hourly = FeatureEngineering(self._stock_data[symbol])
                features_hourly.create_target_features()
                x_hourly, _, _, _, _ = features_hourly.prepare_features()

                # Add stock-specific features to hourly
                x_hourly = self._add_stock_features(x_hourly, symbol)

                # Align hourly with daily
                x_hourly = x_hourly.loc[x_daily.index]

                # Split THIS stock's data by time
                split_idx = int(len(x_daily) * (1 - self.test_size))

                # Train data for this stock
                train_daily_features.append(x_daily.iloc[:split_idx])
                train_hourly_features.append(x_hourly.iloc[:split_idx])
                train_targets.append(y_daily.iloc[:split_idx])
                train_dates.append(pd.Series(dates_daily[:split_idx], index=dates_daily[:split_idx]))
                train_prices.append(prices_daily.iloc[:split_idx])
                train_symbols.extend([symbol] * split_idx)

                # Test data for this stock
                test_daily_features.append(x_daily.iloc[split_idx:])
                test_hourly_features.append(x_hourly.iloc[split_idx:])
                test_targets.append(y_daily.iloc[split_idx:])
                test_dates.append(pd.Series(dates_daily[split_idx:], index=dates_daily[split_idx:]))
                test_prices.append(prices_daily.iloc[split_idx:])
                test_symbols.extend([symbol] * (len(x_daily) - split_idx))

                ConsoleFormatter.success(f"{symbol}: {split_idx} train samples, {len(x_daily) - split_idx} test samples", indentation=1)

            except Exception as e:
                ConsoleFormatter.error(f"Error processing {symbol}: {str(e)}", indentation=1)
                traceback.print_exc()
                continue

        # Combine train data
        ConsoleFormatter.new_section("Combining Training Data", new_lines_before_message=1)

        train_daily = pd.concat(train_daily_features, axis=0, ignore_index=False)
        train_hourly = pd.concat(train_hourly_features, axis=0, ignore_index=False)
        train_targets = pd.concat(train_targets, axis=0, ignore_index=False)
        train_dates = pd.concat(train_dates, axis=0, ignore_index=False)
        train_prices = pd.concat(train_prices, axis=0, ignore_index=False)
        train_symbols_series = pd.Series(train_symbols, index=train_daily.index)

        # Combine test data
        ConsoleFormatter.info("Combining Test Data")
        test_daily = pd.concat(test_daily_features, axis=0, ignore_index=False)
        test_hourly = pd.concat(test_hourly_features, axis=0, ignore_index=False)
        test_targets = pd.concat(test_targets, axis=0, ignore_index=False)
        test_dates = pd.concat(test_dates, axis=0, ignore_index=False)
        test_prices = pd.concat(test_prices, axis=0, ignore_index=False)
        test_symbols_series = pd.Series(test_symbols, index=test_daily.index)

        # Convert categorical columns to category dtype AFTER concatenation
        ConsoleFormatter.info("Converting categorical columns...", new_lines_before_message=1)
        _, train_daily, train_hourly, test_daily, test_hourly = self._create_categorical_features(
            train_daily, train_hourly, test_daily, test_hourly
        )

        ConsoleFormatter.success(f"Total train samples: {len(train_daily)}")
        ConsoleFormatter.success(f"Total test samples: {len(test_daily)}")
        ConsoleFormatter.success(f"Number of stocks: {len(self.symbols)}")
        ConsoleFormatter.success(f"Stocks in test set: {test_symbols_series.nunique()}")
        ConsoleFormatter.success(f"Features per timeframe: {len(train_daily.columns)}")

        self._test_train_data = {
            "train": {
                "daily": train_daily,
                "hourly": train_hourly,
                "targets": train_targets,
                "dates": train_dates,
                "prices": train_prices,
                "symbols": train_symbols_series,
            },
            "test": {
                "daily": test_daily,
                "hourly": test_hourly,
                "targets": test_targets,
                "dates": test_dates,
                "prices": test_prices,
                "symbols": test_symbols_series,
            },
        }

        return self

    def _add_stock_features(self, df: pd.DataFrame, symbol: str):
        """
        Add stock-specific features to the dataframe. These include sector, industry, beta, avg_volume_log, market_cap_log.
        :param df:
        :param symbol:
        :return:
        """
        df["ticker"] = symbol
        df["sector"] = self._downloader.get_sector(symbol)
        df["industry"] = self._downloader.get_industry(symbol)
        df["beta"] = self._downloader.get_beta(symbol)
        df["avg_volume_log"] = np.log(self._downloader.get_avg_volume(symbol) + 1)
        df["market_cap_log"] = np.log(self._downloader.get_market_cap(symbol) + 1)

        return df

    def _create_categorical_features(self, train_daily, train_hourly, test_daily, test_hourly):
        """
        Create category columns so that when it tries to process those columns, it doesn't freak out as they are
        non-numeric.
        :return:
        """
        categorical_cols = ["ticker", "sector", "industry"]
        for col in categorical_cols:
            if col in train_daily.columns:
                train_daily[col] = train_daily[col].astype("category")
                test_daily[col] = test_daily[col].astype("category")
                ConsoleFormatter.info(f"{col}: {train_daily[col].nunique()} unique values", indentation=1)
            if col in train_hourly.columns:
                train_hourly[col] = train_hourly[col].astype("category")
                test_hourly[col] = test_hourly[col].astype("category")

        return self, train_daily, train_hourly, test_daily, test_hourly

    @abstractmethod
    def _create_model(self) -> Any:
        """
        Create the model or models to be trained
        :return:
        """
        pass

    @abstractmethod
    def _optimize_hyperparameters(self) -> Any:
        """
        Run hyperparameter optimization if it is enabled
        :return:
        """
        pass

    @abstractmethod
    def train(self) -> Any:
        """
        Train the model or models provided earlier in this pipeline
        :return:
        """

    @abstractmethod
    def _get_predictions(self):
        """
        Get predictions from the model because the single models expect a raw dataframe and not a dictionary
        :return:
        """
        pass

    def evaluate_model(self):
        """
        Evaluate the model on the test set
        :return:
        """
        if self._model is None:
            raise Exception("Please run train(), before trying to run evaluate_model()")

        ConsoleFormatter.new_section("Per-Stock Performance Analysis", new_lines_before_message=1)

        predictions = self._get_predictions()

        symbols_list = (
            self._symbols_test.tolist() if hasattr(self._symbols_test, "tolist") else list(self._symbols_test)
        )

        unique_symbols = sorted(list(set(symbols_list)))

        ConsoleFormatter.info(f"Evaluating {len(unique_symbols)} stocks...", new_lines_before_message=1)

        self._results = []
        detailed_predictions = []

        for symbol in unique_symbols:
            symbol_mask = [s == symbol for s in symbols_list]
            symbol_indices = [i for i, mask in enumerate(symbol_mask) if mask]

            if len(symbol_indices) == 0:
                continue

            # Extract data for this symbol
            y_true_symbol = self._y_test.iloc[symbol_indices]
            y_pred_symbol = predictions[symbol_indices]
            dates_symbol = self._dates_test.iloc[symbol_indices]
            prices_symbol = self._prices_test.iloc[symbol_indices]

            from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

            mse = mean_squared_error(y_true_symbol, y_pred_symbol)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true_symbol, y_pred_symbol)
            r2 = r2_score(y_true_symbol, y_pred_symbol)

            percentage_errors = np.abs((y_pred_symbol - y_true_symbol) / y_true_symbol) * 100
            mape = percentage_errors.mean()  # Mean Absolute Percentage Error

            direction_actual = np.sign(y_true_symbol)
            direction_pred = np.sign(y_pred_symbol)
            directional_accuracy = (direction_actual == direction_pred).mean() * 100

            start_date = dates_symbol.min()
            end_date = dates_symbol.max()

            self._results.append(
                {
                    "symbol": symbol,
                    "samples": len(symbol_indices),
                    "rmse": rmse,
                    "mae": mae,
                    "mape": mape,
                    "r2": r2,
                    "directional_accuracy": directional_accuracy,
                    "test_start": start_date,
                    "test_end": end_date,
                    "avg_price": prices_symbol.mean(),
                }
            )

            for i, idx in enumerate(symbol_indices):
                detailed_predictions.append(
                    {
                        "symbol": symbol,
                        "date": dates_symbol.iloc[i],
                        "actual_return": y_true_symbol.iloc[i],
                        "predicted_return": y_pred_symbol[i],
                        "price": prices_symbol.iloc[i],
                    }
                )

            ConsoleFormatter().add_newline().apply_bold().add(f"{symbol}:").clear_formatting().build_and_print()
            ConsoleFormatter.info(f"Test Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}", indentation=1)
            ConsoleFormatter.info(f"Samples: {len(symbol_indices)}", indentation=1)
            ConsoleFormatter.info(f"RMSE: {rmse:.6f}", indentation=1)
            ConsoleFormatter.info(f"MAE: {mae:.6f}", indentation=1)
            ConsoleFormatter.info(f"MAPE: {mape:.2f}%", indentation=1)
            ConsoleFormatter.info(f"R²: {r2:.6f}", indentation=1)
            ConsoleFormatter.info(f"Directional Accuracy: {directional_accuracy:.2f}%", indentation=1)

        self._print_summary_stats()

        self._save_results_and_predictions(detailed_predictions)

        return self

    def _print_summary_stats(self):
        ConsoleFormatter.new_section("Summary Statistics Across All Stocks", new_lines_before_message=1)

        results_df = pd.DataFrame(self._results)

        ConsoleFormatter.info(f"Number of stocks evaluated: {len(results_df)}", new_lines_before_message=1)
        ConsoleFormatter.info(f"Average RMSE: {results_df['rmse'].mean():.6f}")
        ConsoleFormatter.info(f"Average MAE: {results_df['mae'].mean():.6f}")
        ConsoleFormatter.info(f"Average MAPE: {results_df['mape'].mean():.2f}%")
        ConsoleFormatter.info(f"Average R²: {results_df['r2'].mean():.6f}")
        ConsoleFormatter.info(f"Average Directional Accuracy: {results_df['directional_accuracy'].mean():.2f}%")
        ConsoleFormatter.info(
            f"Best performing stock (by R²): {results_df.loc[results_df['r2'].idxmax(), 'symbol']} (R²: {results_df['r2'].max():.6f})",
            new_lines_before_message=1
        )
        ConsoleFormatter.info(
            f"Worst performing stock (by R²): {results_df.loc[results_df['r2'].idxmin(), 'symbol']} (R²: {results_df['r2'].min():.6f})"
        )
        ConsoleFormatter.info(
            f"Best directional accuracy: {results_df.loc[results_df['directional_accuracy'].idxmax(), 'symbol']} ({results_df['directional_accuracy'].max():.2f}%)"
        )

    def _save_results_and_predictions(self, detailed_predictions):
        results_df = pd.DataFrame(self._results)
        results_df.to_csv("results/per_stock_performance.csv", index=False)
        ConsoleFormatter.success(f"Per-stock results saved to: results/per_stock_performance.csv", new_lines_before_message=1)

        detailed_df = pd.DataFrame(detailed_predictions)
        detailed_df.to_csv("results/detailed_predictions.csv", index=False)
        ConsoleFormatter.success(f"Detailed predictions saved to: results/detailed_predictions.csv")

    def visualize(self):
        """
        Placeholder for visualization - can be overridden
        :return:
        """
        return self

    def get_model(self):
        """
        Return the trained model
        :return:
        """
        return self._model

    def get_results(self):
        """
        Return evaluation results
        :return:
        """
        return self._results
