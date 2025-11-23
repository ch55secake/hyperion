import traceback

import numpy as np
import pandas as pd

from src.data import StockDataDownloader
from src.feature import FeatureEngineering
from src.lgb import LightGBMStockPredictor
from src.optimise import StockModelOptimizer
from src.stacker import StackedStockPredictor
from src.writer import save_trained_model
from src.xbg import XGBoostStockPredictor


class TrainingPipeline:
    """
    Builder-esq class for training a model.
    I love Java (my-beloved, my lover, my best friend).
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
        self._downloader = None
        self._stock_data = None
        self._symbols_test = None
        self._x_test_dict = None
        self._y_test = None
        self._dates_test = None
        self._prices_test = None
        self._test_train_data = None
        self._model = None
        self._results = None
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

                # Daily features
                features_daily = FeatureEngineering(self._stock_data[symbol])
                features_daily.create_target_features()
                x_daily, y_daily, dates_daily, prices_daily, _ = features_daily.prepare_features()

                # Add stock-specific features
                x_daily["ticker"] = symbol
                x_daily["sector"] = self._downloader.get_sector(symbol)
                x_daily["industry"] = self._downloader.get_industry(symbol)
                x_daily["beta"] = self._downloader.get_beta(symbol)
                x_daily["avg_volume_log"] = np.log(self._downloader.get_avg_volume(symbol) + 1)
                x_daily["market_cap_log"] = np.log(self._downloader.get_market_cap(symbol) + 1)

                # Hourly features
                features_hourly = FeatureEngineering(self._stock_data[symbol])
                features_hourly.create_target_features()
                x_hourly, _, _, _, _ = features_hourly.prepare_features()

                # Add stock-specific features to hourly
                x_hourly["ticker"] = symbol
                x_hourly["sector"] = self._downloader.get_sector(symbol)
                x_hourly["industry"] = self._downloader.get_industry(symbol)
                x_hourly["beta"] = self._downloader.get_beta(symbol)
                x_hourly["avg_volume_log"] = np.log(self._downloader.get_avg_volume(symbol) + 1)
                x_hourly["market_cap_log"] = np.log(self._downloader.get_market_cap(symbol) + 1)

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

                print(f"  ✓ {symbol}: {split_idx} train samples, {len(x_daily) - split_idx} test samples")

            except Exception as e:
                print(f"  ✗ Error processing {symbol}: {str(e)}")
                traceback.print_exc()
                continue

        # Combine train data
        print("\n" + "=" * 60)
        print("Combining Training Data")
        print("=" * 60)

        train_daily = pd.concat(train_daily_features, axis=0, ignore_index=False)
        train_hourly = pd.concat(train_hourly_features, axis=0, ignore_index=False)
        train_targets = pd.concat(train_targets, axis=0, ignore_index=False)
        train_dates = pd.concat(train_dates, axis=0, ignore_index=False)
        train_prices = pd.concat(train_prices, axis=0, ignore_index=False)
        train_symbols_series = pd.Series(train_symbols, index=train_daily.index)

        # Combine test data
        print("Combining Test Data")
        test_daily = pd.concat(test_daily_features, axis=0, ignore_index=False)
        test_hourly = pd.concat(test_hourly_features, axis=0, ignore_index=False)
        test_targets = pd.concat(test_targets, axis=0, ignore_index=False)
        test_dates = pd.concat(test_dates, axis=0, ignore_index=False)
        test_prices = pd.concat(test_prices, axis=0, ignore_index=False)
        test_symbols_series = pd.Series(test_symbols, index=test_daily.index)

        # Convert categorical columns to category dtype AFTER concatenation
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

        print("\n" + "=" * 60)
        print("Training Single Model")
        print("=" * 60)
        print(f"Training samples: {len(x_train_daily)}")
        print(f"Testing samples: {len(x_test_daily)}")
        print(f"Unique stocks in test set: {self._symbols_test.nunique()}")

        if self.should_optimise:
            print("Running hyperparameter optimisation, this will take a while...")
            self._optimise_model(x_train_daily, y_train, x_test_daily, self._y_test)

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

    def _optimise_model(self, x_train_daily=None, y_train=None, x_test_daily=None, y_test=None):
        """
        Run optimization will only be run if the flag is enabled when the pipeline is instantiated.
        :param x_train_daily:
        :param y_train:
        :param x_test_daily:
        :param y_test:
        :return:
        """
        optimizer = StockModelOptimizer(x_train_daily, y_train, x_test_daily, y_test, n_trials=1000, n_jobs=1)
        optimizer.optimize_both()
        optimizer.visualize_studies(save_path="plots/optuna")
        optimizer.save_results(f"params/ALL_STOCKS_best_params.json")

        self._xgb_params, self._lgb_params = optimizer.best_xgb_params, optimizer.best_lgb_params

        return self

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

        predictions = self._model.predict(self._x_test_dict)

        symbols_list = (
            self._symbols_test.tolist() if hasattr(self._symbols_test, "tolist") else list(self._symbols_test)
        )

        unique_symbols = sorted(list(set(symbols_list)))

        print(f"\nEvaluating {len(unique_symbols)} stocks...")

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

            print(f"\n{symbol}:")
            print(f"  Test Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            print(f"  Samples: {len(symbol_indices)}")
            print(f"  RMSE: {rmse:.6f}")
            print(f"  MAE: {mae:.6f}")
            print(f"  MAPE: {mape:.2f}%")
            print(f"  R²: {r2:.6f}")
            print(f"  Directional Accuracy: {directional_accuracy:.2f}%")

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

        results_df.to_csv("results/per_stock_performance.csv", index=False)
        print(f"\n✓ Per-stock results saved to: results/per_stock_performance.csv")

        detailed_df = pd.DataFrame(detailed_predictions)
        detailed_df.to_csv("results/detailed_predictions.csv", index=False)
        print(f"✓ Detailed predictions saved to: results/detailed_predictions.csv")

        return self

    def visualize(self):
        # This doesn't yet produce any plots so it can be blank for now
        return self
