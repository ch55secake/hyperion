import traceback
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.data import StockDataDownloader
from src.feature import FeatureEngineering
from src.feature.regime import classify_regime, COL_REGIME_TREND
from src.util import logger


class BaseTrainingPipeline(ABC):
    """
    Base class for training pipelines
    """

    def __init__(
        self,
        symbols: list[str] | None = None,
        period: str = "2y",
        interval: str = "1d",
        test_size: float = 0.2,
        val_size: float = 0.1,
        should_optimise: bool = False,
        n_trials: int = 1000,
        target_days: int = 10,
        target_horizons: list[int] | None = None,
        target_risk_adjusted: bool = False,
        target_classification: bool = False,
        target_up_threshold: float = 0.02,
        target_down_threshold: float = -0.02,
        r2_save_threshold: float = 0.0012,
        r2_invalid_threshold: float = -0.3,
    ):
        self.symbols = symbols
        self.period = period
        self.interval = interval
        self.test_size = test_size
        self.val_size = val_size
        self.should_optimise = should_optimise
        self.n_trials = n_trials
        self.target_days = target_days
        self.target_horizons = target_horizons
        self.target_risk_adjusted = target_risk_adjusted
        self.target_classification = target_classification
        self.target_up_threshold = target_up_threshold
        self.target_down_threshold = target_down_threshold
        self.r2_save_threshold = r2_save_threshold
        self.r2_invalid_threshold = r2_invalid_threshold

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
        self._test_results = None
        self._split_idx = None

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
            logger.info(f"Read {len(symbols)} tickers from resources/tickers.txt")

        return self

    def download_data(self):
        """
        Download the historical stock data for all tickers provided in the resources/tickers.txt
        :return: pipeline instance but should also update the _stock_data attribute
        """
        logger.info("=" * 60)
        logger.info("Downloading and Preparing Data for All Stocks")
        logger.info("=" * 60)

        if self.symbols is None:
            raise Exception("Please run read_tickers(), before trying to run download_data()")

        self._downloader = StockDataDownloader(self.symbols, period=self.period, interval=self.interval)
        self._stock_data, failed = self._downloader.download_data()

        if failed:
            logger.warning("%d symbol(s) failed to download and will be excluded: %s", len(failed), failed)

        if not self._stock_data:
            logger.warning("No data downloaded. Exiting.")
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
                logger.info(f"Processing {symbol}...")

                features_daily = FeatureEngineering(self._stock_data[symbol])
                features_daily.create_target_features(
                    target_days=self.target_days,
                    horizons=self.target_horizons,
                    risk_adjusted=self.target_risk_adjusted,
                    classification=self.target_classification,
                    up_threshold=self.target_up_threshold,
                    down_threshold=self.target_down_threshold,
                )
                x_daily, y_daily, dates_daily, prices_daily, _ = features_daily.prepare_features()

                x_daily = self._add_stock_features(x_daily, symbol)

                x_hourly = x_daily.copy()

                self._split_idx = int(len(x_daily) * (1 - self.test_size))

                self._log_regime_coverage(prices_daily, self._split_idx, symbol)

                train_daily_features.append(x_daily.iloc[: self._split_idx])
                train_hourly_features.append(x_hourly.iloc[: self._split_idx])
                train_targets.append(y_daily.iloc[: self._split_idx])
                train_dates.append(pd.Series(dates_daily[: self._split_idx], index=dates_daily[: self._split_idx]))
                train_prices.append(prices_daily.iloc[: self._split_idx])
                train_symbols.extend([symbol] * self._split_idx)

                test_daily_features.append(x_daily.iloc[self._split_idx :])
                test_hourly_features.append(x_hourly.iloc[self._split_idx :])
                test_targets.append(y_daily.iloc[self._split_idx :])
                test_dates.append(pd.Series(dates_daily[self._split_idx :], index=dates_daily[self._split_idx :]))
                test_prices.append(prices_daily.iloc[self._split_idx :])
                test_symbols.extend([symbol] * (len(x_daily) - self._split_idx))

                train_end = dates_daily[self._split_idx - 1] if self._split_idx > 0 else "N/A"
                test_start = dates_daily[self._split_idx] if self._split_idx < len(dates_daily) else "N/A"
                logger.info(
                    f"{symbol}: {self._split_idx} train samples (up to {train_end}), "
                    f"{len(x_daily) - self._split_idx} test samples (from {test_start})"
                )

            except Exception as e:
                logger.error(f"Error processing {symbol}: {str(e)}")
                traceback.print_exc()
                continue

        logger.info("=" * 60)
        logger.info("Combining Training Data")
        logger.info("=" * 60)

        train_daily, train_hourly, train_targets, train_dates, train_prices = self._combine_features(
            daily_features=train_daily_features,
            hourly_features=train_hourly_features,
            train_targets=train_targets,
            train_dates=train_dates,
            train_prices=train_prices,
        )

        train_symbols_series = pd.Series(train_symbols, index=train_daily.index)

        logger.info("Combining Test Data")
        test_daily, test_hourly, test_targets, test_dates, test_prices = self._combine_features(
            daily_features=test_daily_features,
            hourly_features=test_hourly_features,
            train_targets=test_targets,
            train_dates=test_dates,
            train_prices=test_prices,
        )

        test_symbols_series = pd.Series(test_symbols, index=test_daily.index)

        logger.info("Converting categorical columns...")
        _, train_daily, train_hourly, test_daily, test_hourly = self._create_categorical_features(
            train_daily, train_hourly, test_daily, test_hourly
        )

        logger.info(f"Total train samples: {len(train_daily)}")
        logger.info(f"Total test samples: {len(test_daily)}")
        logger.info(f"Number of stocks: {len(self.symbols)}")
        logger.info(f"Stocks in test set: {test_symbols_series.nunique()}")
        logger.info(f"Features per timeframe: {len(train_daily.columns)}")

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
        Add stock-specific features to the dataframe.
        These include sector, industry, beta, avg_volume_log, market_cap_log.
        :param df:
        :param symbol:
        :return:
        """
        df["ticker"] = symbol
        df["sector"] = self._downloader.get_sector(symbol)
        df["industry"] = self._downloader.get_industry(symbol)
        df["beta"] = self._downloader.get_beta(symbol)
        df["avg_volume_log"] = np.log(self._downloader.get_avg_volume(symbol) + 1)
        raw_market_cap = self._downloader.get_market_cap(symbol)
        df["market_cap_log"] = np.log(raw_market_cap + 1) if raw_market_cap is not None else np.nan

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
                logger.debug(f"{col}: {train_daily[col].nunique()} unique values")
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

    @abstractmethod
    def simulate(self, initial_capital: float = 10000, transaction_cost: float = 0.001) -> Any:
        """
        Simulate trading day by day based on the model predictions
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

    @staticmethod
    def _log_regime_coverage(prices: pd.Series, split_idx: int, symbol: str) -> None:
        """Log regime coverage for train and test splits to aid in evaluation.

        Warns when the test split contains only a single regime, which can make
        performance metrics misleadingly optimistic.

        Parameters
        ----------
        prices:
            Full closing-price series for the symbol (chronologically sorted).
        split_idx:
            Index position separating the training set (before) from the test
            set (from this index onwards).
        symbol:
            Ticker used in log messages.
        """
        if split_idx <= 0 or split_idx >= len(prices):
            return

        returns = prices.pct_change(1).fillna(0)
        regime_df = classify_regime(prices, returns)
        trend = regime_df[COL_REGIME_TREND]

        train_regimes = set(trend.iloc[:split_idx].unique())
        test_regimes = set(trend.iloc[split_idx:].unique())

        regime_names = {0: "bull", 1: "bear", 2: "sideways"}
        train_labels = {regime_names.get(r, str(r)) for r in train_regimes}
        test_labels = {regime_names.get(r, str(r)) for r in test_regimes}

        logger.debug("%s — train regimes: %s | test regimes: %s", symbol, train_labels, test_labels)

        if len(test_regimes) < 2:
            logger.warning(
                "%s — test split contains only '%s' regime(s). "
                "Consider a longer test period for more robust evaluation.",
                symbol,
                test_labels,
            )

    def evaluate_model(self):
        """
        Evaluate the model on the test set
        :return:
        """
        if self._model is None:
            raise Exception("Please run train(), before trying to run evaluate_model()")

        logger.info("=" * 60)
        logger.info("Per-Stock Performance Analysis")
        logger.info("=" * 60)

        predictions = self._get_predictions()

        y_test_reset = self._y_test.reset_index(drop=True)
        dates_reset = self._dates_test.reset_index(drop=True)
        prices_reset = self._prices_test.reset_index(drop=True)
        symbols_reset = self._symbols_test.reset_index(drop=True)

        unique_symbols = sorted(symbols_reset.unique())

        logger.info(f"Evaluating {len(unique_symbols)} stocks...")
        logger.debug(f"Symbols in test data: {unique_symbols[:5]}...")
        logger.debug(f"Total test samples: {len(symbols_reset)}")
        logger.debug(f"Total predictions: {len(predictions)}")

        self._results = []
        detailed_predictions = []

        for symbol in unique_symbols:
            logger.debug(f"Processing symbol: {symbol}")
            symbol_mask = symbols_reset == symbol

            if symbol_mask.sum() == 0:
                logger.warning(f"No data found for {symbol}")
                continue
            else:
                logger.debug(f"Found {symbol_mask.sum()} samples for {symbol}")

            y_true_symbol = y_test_reset[symbol_mask]
            y_pred_symbol = predictions[symbol_mask]
            dates_symbol = dates_reset[symbol_mask]
            prices_symbol = prices_reset[symbol_mask]

            mse = mean_squared_error(y_true_symbol, y_pred_symbol)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true_symbol, y_pred_symbol)
            r2 = r2_score(y_true_symbol, y_pred_symbol)

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

        self._print_summary_stats()

        self._save_results_and_predictions(detailed_predictions)

        return self

    def _print_summary_stats(self):
        logger.info("=" * 60)
        logger.info("Summary Statistics Across All Stocks")
        logger.info("=" * 60)

        results_df = pd.DataFrame(self._results)

        logger.info(f"Number of stocks evaluated: {len(results_df)}")

        if len(results_df) == 0:
            logger.warning("No stocks were evaluated. Check test data and symbols.")
            return

        logger.info(f"Average RMSE: {results_df['rmse'].mean():.6f}")
        logger.info(f"Average MAE: {results_df['mae'].mean():.6f}")
        logger.info(f"Average R\u00b2: {results_df['r2'].mean():.6f}")
        logger.info(f"Average Directional Accuracy: {results_df['directional_accuracy'].mean():.2f}%")
        logger.info(
            "Best performing stock (by R²): %s (R²: %.6f)",
            results_df.loc[results_df["r2"].idxmax(), "symbol"],
            results_df["r2"].max(),
        )
        logger.info(
            "Worst performing stock (by R²): %s (R²: %.6f)",
            results_df.loc[results_df["r2"].idxmin(), "symbol"],
            results_df["r2"].min(),
        )
        logger.info(
            "Best directional accuracy: %s (%.2f%%)",
            results_df.loc[results_df["directional_accuracy"].idxmax(), "symbol"],
            results_df["directional_accuracy"].max(),
        )

    def _save_results_and_predictions(self, detailed_predictions):
        results_df = pd.DataFrame(self._results)
        results_df.to_csv("results/per_stock_performance.csv", index=False)
        logger.info("Per-stock results saved to: results/per_stock_performance.csv")

        detailed_df = pd.DataFrame(detailed_predictions)
        detailed_df.to_csv("results/detailed_predictions.csv", index=False)
        logger.info("Detailed predictions saved to: results/detailed_predictions.csv")

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
