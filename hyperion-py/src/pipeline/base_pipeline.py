import traceback
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

from src.data import StockDataDownloader
from src.feature import FeatureEngineering


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

        self._downloader = None
        self._stock_data = None
        self._model = None
        self._results = None
        self._test_train_data = None

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
            print(f"Read {len(symbols)} tickers from resources/tickers.txt")

        return self

    def download_data(self):
        """
        Download the historical stock data for all tickers provided in the resources/tickers.txt
        :return: pipeline instance but should also update the _stock_data attribute
        """
        print("\n" + "=" * 60)
        print("Downloading and Preparing Data for All Stocks")
        print("=" * 60)

        if self.symbols is None:
            raise Exception("Please run read_tickers(), before trying to run download_data()")

        self._downloader = StockDataDownloader(self.symbols, period=self.period, interval=self.interval)
        self._stock_data = self._downloader.download_data()

        if not self._stock_data:
            print("⚠️  No data downloaded. Exiting.")
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
                print(f"\nProcessing {symbol}...")

                features_daily = FeatureEngineering(self._stock_data[symbol])
                features_daily.create_target_features()
                x_daily, y_daily, dates_daily, prices_daily, _ = features_daily.prepare_features()

                x_daily = self._add_stock_features(x_daily, symbol)

                features_hourly = FeatureEngineering(self._stock_data[symbol])
                features_hourly.create_target_features()
                x_hourly, _, _, _, _ = features_hourly.prepare_features()

                x_hourly = self._add_stock_features(x_hourly, symbol)

                x_hourly = x_hourly.loc[x_daily.index]

                split_idx = int(len(x_daily) * (1 - self.test_size))

                train_daily_features.append(x_daily.iloc[:split_idx])
                train_hourly_features.append(x_hourly.iloc[:split_idx])
                train_targets.append(y_daily.iloc[:split_idx])
                train_dates.append(pd.Series(dates_daily[:split_idx], index=dates_daily[:split_idx]))
                train_prices.append(prices_daily.iloc[:split_idx])
                train_symbols.extend([symbol] * split_idx)

                test_daily_features.append(x_daily.iloc[split_idx:])
                test_hourly_features.append(x_hourly.iloc[split_idx:])
                test_targets.append(y_daily.iloc[split_idx:])
                test_dates.append(pd.Series(dates_daily[split_idx:], index=dates_daily[split_idx:]))
                test_prices.append(prices_daily.iloc[split_idx:])
                test_symbols.extend([symbol] * (len(x_daily) - split_idx))

                print(f"  ✓ {symbol}: {split_idx} train samples, {len(x_daily) - split_idx} test samples")

            except Exception as e:
                print(f"  ✗ Error processing {symbol}: {str(e)}")
                traceback.print_exc()
                continue

        print("\n" + "=" * 60)
        print("Combining Training Data")
        print("=" * 60)

        train_daily, train_hourly, train_targets, train_dates, train_prices = self._combine_features(
            daily_features=train_daily_features,
            hourly_features=train_hourly_features,
            train_targets=train_targets,
            train_dates=train_dates,
            train_prices=train_prices,
        )

        train_symbols_series = pd.Series(train_symbols, index=train_daily.index)

        print("Combining Test Data")
        test_daily, test_hourly, test_targets, test_dates, test_prices = self._combine_features(
            daily_features=test_daily_features,
            hourly_features=test_hourly_features,
            train_targets=test_targets,
            train_dates=test_dates,
            train_prices=test_prices,
        )

        test_symbols_series = pd.Series(test_symbols, index=test_daily.index)

        print("\nConverting categorical columns...")
        _, train_daily, train_hourly, test_daily, test_hourly = self._create_categorical_features(
            train_daily, train_hourly, test_daily, test_hourly
        )

        print(f"✓ Total train samples: {len(train_daily)}")
        print(f"✓ Total test samples: {len(test_daily)}")
        print(f"✓ Number of stocks: {len(self.symbols)}")
        print(f"✓ Stocks in test set: {test_symbols_series.nunique()}")
        print(f"✓ Features per timeframe: {len(train_daily.columns)}")

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
                print(f"  {col}: {train_daily[col].nunique()} unique values")
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
        pass

    @abstractmethod
    def _get_predictions(self):
        """
        Get predictions from the model because the single models expect a raw dataframe and not a dictionary
        :return:
        """
        pass

    def _populate_test_train_data(self):
        """
        Populate test train data to avoid duplication between pipelines
        :return:
        """
        self._y_test = self._test_train_data["test"]["targets"]
        self._dates_test = self._test_train_data["test"]["dates"]
        self._prices_test = self._test_train_data["test"]["prices"]
        self._symbols_test = self._test_train_data["test"]["symbols"]

    @staticmethod
    def _combine_features(daily_features, hourly_features, train_targets, train_dates, train_prices):
        """
        Concatenate dataframes so that they are along the same axis
        :return:
        """
        train_daily = pd.concat(daily_features, axis=0, ignore_index=False)
        train_hourly = pd.concat(hourly_features, axis=0, ignore_index=False)
        train_targets = pd.concat(train_targets, axis=0, ignore_index=False)
        train_dates = pd.concat(train_dates, axis=0, ignore_index=False)
        train_prices = pd.concat(train_prices, axis=0, ignore_index=False)

        return train_daily, train_hourly, train_targets, train_dates, train_prices

    def evaluate_model(self):
        """
        Evaluate the model on the test set
        :return:
        """
        if self._model is None:
            raise Exception("Please run train(), before trying to run evaluate_model()")

        print("\n" + "=" * 60)
        print("Per-Stock Performance Analysis")
        print("=" * 60)

        predictions = self._get_predictions()

        y_test_reset = self._y_test.reset_index(drop=True)
        dates_reset = self._dates_test.reset_index(drop=True)
        prices_reset = self._prices_test.reset_index(drop=True)
        symbols_reset = self._symbols_test.reset_index(drop=True)

        unique_symbols = sorted(symbols_reset.unique())

        print(f"\nEvaluating {len(unique_symbols)} stocks...")

        self._results = []
        detailed_predictions = []

        for symbol in unique_symbols:
            symbol_mask = symbols_reset == symbol

            if symbol_mask.sum() == 0:
                continue

            y_true_symbol = y_test_reset[symbol_mask]
            y_pred_symbol = predictions[symbol_mask]
            dates_symbol = dates_reset[symbol_mask]
            prices_symbol = prices_reset[symbol_mask]

            from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

            mse = mean_squared_error(y_true_symbol, y_pred_symbol)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true_symbol, y_pred_symbol)
            r2 = r2_score(y_true_symbol, y_pred_symbol)

            percentage_errors = np.abs((y_pred_symbol - y_true_symbol) / y_true_symbol) * 100
            mape = percentage_errors.mean()

            direction_actual = np.sign(y_true_symbol)
            direction_pred = np.sign(y_pred_symbol)
            directional_accuracy = (direction_actual == direction_pred).mean() * 100

            start_date = dates_symbol.min()
            end_date = dates_symbol.max()

            self._results.append(
                {
                    "symbol": symbol,
                    "samples": symbol_mask.sum(),
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

            y_pred_array = np.array(y_pred_symbol)
            symbol_indices = np.where(symbol_mask)[0]
            for i, idx in enumerate(symbol_indices):
                detailed_predictions.append(
                    {
                        "symbol": symbol,
                        "date": dates_symbol.iloc[i],
                        "actual_return": y_true_symbol.iloc[i],
                        "predicted_return": y_pred_array[i],
                        "price": prices_symbol.iloc[i],
                    }
                )

            print(f"\n{symbol}:")
            print(f"  Test Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            print(f"  Samples: {symbol_mask.sum()}")
            print(f"  RMSE: {rmse:.6f}")
            print(f"  MAE: {mae:.6f}")
            print(f"  MAPE: {mape:.2f}%")
            print(f"  R²: {r2:.6f}")
            print(f"  Directional Accuracy: {directional_accuracy:.2f}%")

        self._print_summary_stats()

        self._save_results_and_predictions(detailed_predictions)

        return self

    def _print_summary_stats(self):
        print("\n" + "=" * 60)
        print("Summary Statistics Across All Stocks")
        print("=" * 60)

        results_df = pd.DataFrame(self._results)

        print(f"\nNumber of stocks evaluated: {len(results_df)}")
        print(f"Average RMSE: {results_df['rmse'].mean():.6f}")
        print(f"Average MAE: {results_df['mae'].mean():.6f}")
        print(f"Average MAPE: {results_df['mape'].mean():.2f}%")
        print(f"Average R²: {results_df['r2'].mean():.6f}")
        print(f"Average Directional Accuracy: {results_df['directional_accuracy'].mean():.2f}%")
        print(
            f"\nBest performing stock (by R²): {results_df.loc[results_df['r2'].idxmax(), 'symbol']} (R²: {results_df['r2'].max():.6f})"
        )
        print(
            f"Worst performing stock (by R²): {results_df.loc[results_df['r2'].idxmin(), 'symbol']} (R²: {results_df['r2'].min():.6f})"
        )
        print(
            f"Best directional accuracy: {results_df.loc[results_df['directional_accuracy'].idxmax(), 'symbol']} ({results_df['directional_accuracy'].max():.2f}%)"
        )

    def _save_results_and_predictions(self, detailed_predictions):
        results_df = pd.DataFrame(self._results)
        results_df.to_csv("results/per_stock_performance.csv", index=False)
        print("\n✓ Per-stock results saved to: results/per_stock_performance.csv")

        detailed_df = pd.DataFrame(detailed_predictions)
        detailed_df.to_csv("results/detailed_predictions.csv", index=False)
        print("✓ Detailed predictions saved to: results/detailed_predictions.csv")

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
