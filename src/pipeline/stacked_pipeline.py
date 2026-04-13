import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from typing_extensions import override

from src.align import align_targets_across_intervals, ensure_prediction_alignment
from src.data import StockDataDownloader
from src.feature import FeatureEngineering
from src.model import LightGBMStockPredictor
from src.model import StackedStockPredictor
from src.model import XGBoostStockPredictor
from src.optimise import StockModelOptimizer
from src.pipeline.base_pipeline import BaseTrainingPipeline
from src.ranking import StockRanker
from src.simulation import TradingSimulator
from src.simulation.strategy.strategy_registry import StrategyRegistry
from src.util import logger
from src.writer import save_trained_model
from src.feature.feature_split import FeaturePartition, derive_feature_split

# Required for the usage of the strategy registry


def _simulate_ticker_worker(
    ticker_df: pd.DataFrame,
    strategy_key: str,
    initial_capital: int,
    transaction_cost: float = 0.001,
) -> tuple[str, dict | None, str | None]:
    """Simulate a single ticker; designed for parallel execution via ProcessPoolExecutor.

    Returns a 3-tuple ``(symbol, results, skip_reason)`` where:
    - *results* is the simulation result dict on success, or ``None`` when the
      ticker is skipped.
    - *skip_reason* describes why the ticker was skipped; ``None`` on success.

    Unexpected errors are not caught here and will propagate to the caller via
    ``Future.result()``.
    """
    symbol = str(ticker_df["symbol"].iloc[0])
    strategy_class = StrategyRegistry.get(strategy_key)

    if len(ticker_df) < strategy_class.get_minimum_data_points():
        return symbol, None, f"insufficient data ({len(ticker_df)} rows)"

    additional_data = strategy_class.get_extra_params(ticker_df.set_index("date")["price"])

    simulator = TradingSimulator(initial_capital=initial_capital, transaction_cost=transaction_cost)
    strategy = StrategyRegistry.create(
        name=strategy_key, simulator=simulator, capital=initial_capital, **additional_data
    )

    ticker_data_reset = ticker_df.reset_index(drop=True)

    results = simulator.simulate(
        predictions=ticker_data_reset["prediction"],
        actual_returns=ticker_data_reset["actual_return"],
        prices=ticker_data_reset["price"],
        dates=ticker_data_reset["date"],
        strategy=strategy,
        threshold="auto",
    )

    return symbol, results, None


class StackedModelTrainingPipeline(BaseTrainingPipeline):

    def __init__(
        self,
        intervals: list[str],
        interval_roles: dict[str, str] | None = None,
        short_term_threshold: int = 20,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.intervals = intervals
        self._xgb_params = None
        self._lgb_params = None

        self.default_interval = self.intervals[0]
        self.short_term_threshold = short_term_threshold
        self.interval_roles = dict(interval_roles) if interval_roles is not None else {}
        self.interval_roles.setdefault(self.default_interval, "daily")
        for interval in self.intervals:
            self.interval_roles.setdefault(interval, "hourly")

        self.interval_feature_sets: dict[str, list[str]] = {}
        self.feature_partitions: dict[str, FeaturePartition] = {}

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
        Download both hourly and daily data instead of populating just daily data.
        All intervals are downloaded concurrently.
        :return:
        """
        logger.info("=" * 60)
        logger.info("Downloading and Preparing Data for All Stocks")
        logger.info("=" * 60)

        if self.symbols is None:
            raise Exception("Please run read_tickers(), before trying to run download_data()")

        self._stock_data = {}

        def _download_interval(interval):
            downloader = StockDataDownloader(self.symbols, period=self.period, interval=interval)
            interval_data, failed = downloader.download_data()
            if failed:
                logger.warning(
                    "%d symbol(s) failed to download for interval '%s' and will be excluded: %s",
                    len(failed),
                    interval,
                    failed,
                )
            return interval, downloader, interval_data

        interval_downloaders: dict[str, StockDataDownloader] = {}
        with ThreadPoolExecutor(max_workers=len(self.intervals)) as executor:
            futures = {executor.submit(_download_interval, interval): interval for interval in self.intervals}
            for future in as_completed(futures):
                interval, downloader, data = future.result()
                self._stock_data[interval] = data
                interval_downloaders[interval] = downloader

        self._downloader = interval_downloaders[self.intervals[0]]

        if not self._stock_data:
            logger.warning("No data downloaded. Exiting.")
            return None

        return self

    def __create_categorical_features(self, train, test):
        """
        Create categorical features for specified columns in the dataset.
        :param train: train split of the data
        :param test: test split of the data
        :return: train and test
        """
        categorical_cols: list[str] = ["ticker", "sector", "industry"]
        for interval in self.intervals:
            for col in categorical_cols:
                if col in train[interval].columns:
                    train[interval][col] = train[interval][col].astype("category")
                    test[interval][col] = test[interval][col].astype("category")
                    logger.debug(f"{interval} {col}: {train[interval][col].nunique()} unique values")

        return self, train, test

    def _engineer_features_for_symbol(self, symbol: str, data: pd.DataFrame):
        """
        Run feature engineering for a single symbol and return split train/test data.
        Designed to be called in parallel via joblib.
        :param symbol: ticker symbol
        :param data: raw OHLCV DataFrame for this symbol and interval
        :return: tuple of (symbol, x, y, dates, prices, split_idx) or (symbol, None, ...) on error
        """
        try:
            logger.info(f"Processing {symbol}...")
            features = FeatureEngineering(data)
            features.create_target_features(target_days=self.target_days)
            x, y, dates, prices, _ = features.prepare_features()
            x = self._add_stock_features(x, symbol)
            split_idx = int(len(x) * (1 - self.test_size))
            logger.info(f"{symbol}: {split_idx} train samples, {len(x) - split_idx} test samples")
            return symbol, x, y, dates, prices, split_idx
        except Exception as e:
            logger.error(f"Error processing {symbol}: {str(e)}")
            traceback.print_exc()
            return symbol, None, None, None, None, None

    @override
    def prepare_features(self):
        """
        Prepare all features for both daily and hourly data, as well as the test and train split.
        Feature engineering is parallelized across tickers using joblib threads.
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
            available_symbols = [s for s in self.symbols if s in self._stock_data[interval]]
            if len(available_symbols) < len(self.symbols):
                missing = set(self.symbols) - set(available_symbols)
                logger.warning(
                    "Skipping %d symbol(s) missing from '%s' interval data: %s",
                    len(missing),
                    interval,
                    missing,
                )
            results = Parallel(n_jobs=-1, prefer="threads")(
                delayed(self._engineer_features_for_symbol)(symbol, self._stock_data[interval][symbol])
                for symbol in available_symbols
            )

            for symbol, x, y, dates, prices, split_idx in results:
                if x is None:
                    continue

                self._split_idx = split_idx

                train_features[interval].append(x.iloc[:split_idx])
                train_targets[interval].append(y.iloc[:split_idx])
                train_dates[interval].append(pd.Series(dates[:split_idx], index=dates[:split_idx]))
                train_prices[interval].append(prices.iloc[:split_idx])
                train_symbols[interval].extend([symbol] * split_idx)

                test_features[interval].append(x.iloc[split_idx:])
                test_targets[interval].append(y.iloc[split_idx:])
                test_dates[interval].append(pd.Series(dates[split_idx:], index=dates[split_idx:]))
                test_prices[interval].append(prices.iloc[split_idx:])
                test_symbols[interval].extend([symbol] * (len(x) - split_idx))

        logger.info("=" * 60)
        logger.info("Combining Training Data")
        logger.info("=" * 60)
        train_intervals = dict()
        for interval in self.intervals:
            train_intervals[interval] = pd.concat(train_features[interval], axis=0, ignore_index=False)

        train_targets_dict = dict()
        for interval in self.intervals:
            train_targets_dict[interval] = pd.concat(train_targets[interval], axis=0, ignore_index=False)

        train_dates = pd.concat(train_dates[self.default_interval], axis=0, ignore_index=False)
        train_prices = pd.concat(train_prices[self.default_interval], axis=0, ignore_index=False)

        combined_train_symbols = train_symbols[self.default_interval]
        train_symbols_series = pd.Series(combined_train_symbols, index=train_intervals[self.default_interval].index)

        logger.info("Combining Test Data")
        test_intervals = dict()
        for interval in self.intervals:
            test_intervals[interval] = pd.concat(test_features[interval], axis=0, ignore_index=False)

        test_targets_dict = dict()
        for interval in self.intervals:
            test_targets_dict[interval] = pd.concat(test_targets[interval], axis=0, ignore_index=False)

        test_dates = pd.concat(test_dates[self.default_interval], axis=0, ignore_index=False)
        test_prices = pd.concat(test_prices[self.default_interval], axis=0, ignore_index=False)

        test_symbols_series = pd.Series(
            test_symbols[self.default_interval], index=test_intervals[self.default_interval].index
        )

        logger.info("Converting categorical columns...")
        _, train_intervals, test_intervals = self.__create_categorical_features(train_intervals, test_intervals)

        logger.info(f"Total train samples: {len(train_intervals[self.default_interval])}")
        logger.info(f"Total test samples: {len(test_intervals[self.default_interval])}")
        logger.info(f"Number of stocks: {len(self.symbols)}")
        logger.info(f"Stocks in test set: {test_symbols_series.nunique()}")
        logger.info(f"Features per timeframe: {len(train_intervals[self.default_interval].columns)}")

        self._test_train_data = {
            "train": {
                **{interval: train_intervals[interval] for interval in self.intervals},
                "targets": train_targets_dict,
                "dates": train_dates,
                "prices": train_prices,
                "symbols": train_symbols_series,
            },
            "test": {
                **{interval: test_intervals[interval] for interval in self.intervals},
                "targets": test_targets_dict,
                "dates": test_dates,
                "prices": test_prices,
                "symbols": test_symbols_series,
            },
        }

        self.feature_partitions = {}
        self.interval_feature_sets = {}
        for interval in self.intervals:
            partition = derive_feature_split(
                train_intervals[interval].columns.tolist(), short_term_threshold=self.short_term_threshold
            )
            self.feature_partitions[interval] = partition
            role = self.interval_roles.get(interval, "hourly")
            selected_columns = set(partition.get("shared", []))
            selected_columns.update(partition.get(role, []))
            self.interval_feature_sets[interval] = sorted(selected_columns)

        return self

    def describe_feature_splits(self) -> dict[str, dict[str, int | str]]:
        """
        Return feature counts broken down by role, shared, and total.
        :return: Summary dictionary with feature counts
        """
        summary: dict[str, dict[str, int | str]] = {}
        for interval, columns in self.interval_feature_sets.items():
            role = self.interval_roles.get(interval, "hourly")
            partition = self.feature_partitions.get(interval, {"daily": [], "hourly": [], "shared": []})
            summary[interval] = {
                "role": role,
                "role_specific": len(partition.get(role, [])),
                "shared": len(partition.get("shared", [])),
                "total": len(columns),
            }
        return summary

    def _select_interval_features(self, interval: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Restrict a frame to the columns assigned to the interval.
        :return: Restricted DataFrame with interval-specific features
        """
        columns = self.interval_feature_sets.get(interval)
        if not columns:
            return df
        return df.loc[:, columns]

    def _log_feature_split_summary(self) -> None:
        logger.info("Feature split per interval:")
        for interval, info in self.describe_feature_splits().items():
            logger.info(
                f"  {interval} ({info['role']}): {info['role_specific']} {info['role']} + {info['shared']} shared = {info['total']} total"
            )

    def _populate_test_train_data(self):
        """
        Populate test train data with support for multiple intervals
        Override base method to handle interval-specific targets
        :return:
        """
        if isinstance(self._test_train_data["test"]["targets"], dict):
            self._y_test_dict = self._test_train_data["test"]["targets"]
            self._y_test = self._y_test_dict[self.default_interval]

        self._dates_test = self._test_train_data["test"]["dates"]
        self._prices_test = self._test_train_data["test"]["prices"]
        self._symbols_test = self._test_train_data["test"]["symbols"]

    def _validate_data_consistency(self):
        """
        Validate that all test data components are properly aligned.
        :return: list of validation results
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

            if validation["lengths_match"] and x_test_len > 0:
                x_index = self._test_train_data["test"][interval].index
                y_index = self._y_test_dict[interval].index
                validation["indices_equal"] = x_index.equals(y_index)

            validations.append(validation)

        logger.info("=" * 60)
        logger.info("Data Consistency Validation")
        logger.info("=" * 60)
        for val in validations:
            logger.debug(f"Interval: {val['interval']}")
            logger.debug(f"  X_test length: {val['x_test_length']}")
            logger.debug(f"  Y_test length: {val['y_test_length']}")
            logger.debug(f"  Lengths match: {val['lengths_match']}")
            if "indices_equal" in val:
                logger.debug(f"  Indices equal: {val['indices_equal']}")

        return validations

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
        :return:
        """
        if self._test_train_data is None:
            raise Exception("Please run prepare_features(), before trying to run train()")

        x_train = dict()
        x_test = dict()

        for interval in self.intervals:
            x_train[interval] = self._test_train_data["train"][interval]
            x_test[interval] = self._test_train_data["test"][interval]

        x_train_selected = {
            interval: self._select_interval_features(interval, x_train[interval]) for interval in self.intervals
        }
        x_test_selected = {
            interval: self._select_interval_features(interval, x_test[interval]) for interval in self.intervals
        }

        if isinstance(self._test_train_data["train"]["targets"], dict):
            y_train_dict = self._test_train_data["train"]["targets"]

        self._populate_test_train_data()

        self._validate_data_consistency()

        self._log_feature_split_summary()

        logger.info("=" * 60)
        logger.info("Training Stacked Model")
        logger.info("=" * 60)
        logger.info(f"Training samples: {len(x_train[self.default_interval])}")
        logger.info(f"Testing samples: {len(x_test[self.default_interval])}")
        logger.info(f"Unique stocks in test set: {self._symbols_test.nunique()}")

        if self.should_optimise:
            logger.info("Running hyperparameter optimisation, this will take a while...")
            # self._optimize_hyperparameters(x_train_daily, y_train, x_test_daily, self._y_test)

        self._model = StackedStockPredictor(
            {
                "1d": XGBoostStockPredictor(params=self._xgb_params),
                "1h": LightGBMStockPredictor(params=self._lgb_params),
            }
        )

        train_data = {
            interval: (
                x_train_selected[interval],
                y_train_dict[interval],
                x_test_selected[interval],
                self._y_test_dict[interval],
            )
            for interval in self.intervals
        }

        self._model.train(train_data)

        self._x_test_dict = {interval: x_test_selected[interval] for interval in self.intervals}

        aligned_targets = align_targets_across_intervals(self._y_test_dict, self.default_interval, self.intervals)
        self._test_results = self._model.evaluate(self._x_test_dict, aligned_targets[self.default_interval])

        model_name: str = "ALL_STOCKS"
        save_trained_model(
            self._model, model_name, self._test_results, self.r2_save_threshold, self.r2_invalid_threshold
        )

        return self

    def simulate(
        self, initial_capital: float = 10000, transaction_cost: float = 0.001, tickers=None, strategy_name: str = None
    ):
        """
        Use the stacked trained model to simulate trading day by day, per ticker
        :return:
        """
        if tickers is None:
            tickers = ["AAPL"]
        predictions = self._test_results.get("predictions")
        if predictions is None:
            logger.warning("Predictions missing, computing via predictor.predict()")
            predictions = self._get_predictions()

        min_len = min(
            len(predictions), len(self._y_test), len(self._symbols_test), len(self._dates_test), len(self._prices_test)
        )

        if len(predictions) != min_len:
            logger.warning(f"Truncating all test data to {min_len} samples for alignment")

        pred_len = len(predictions)

        aligned_y_test = ensure_prediction_alignment(predictions, self._y_test)

        test_data = {
            "prediction": predictions[:pred_len],
        }

        test_data = self.validate_data_required_for_simulation(pred_len, test_data)

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
            logger.info("=" * 60)
            logger.info(f"Strategy: {strategy_key}")
            logger.info("=" * 60)

            strategy_results = {}

            ticker_dfs = [test_df[test_df["symbol"] == symbol].sort_values("date") for symbol in unique_symbols]

            with ProcessPoolExecutor() as pool:
                futures = [
                    (
                        pool.submit(_simulate_ticker_worker, df, strategy_key, int(initial_capital), transaction_cost),
                        str(df["symbol"].iloc[0]),
                    )
                    for df in ticker_dfs
                ]

                for future, symbol in futures:
                    try:
                        sym, results, skip_reason = future.result()
                        if results is None:
                            logger.warning(f"Skipping {sym}: {skip_reason}")
                        else:
                            strategy_results[sym] = results
                            logger.info(f"--- {sym} ({len(results['portfolio_history'])} samples) ---")
                            logger.info(f"Final Value: ${results['final_value']:,.2f}")
                            logger.info(f"Return: {results['total_return'] * 100:.2f}%")
                    except Exception:
                        logger.exception(f"Error running {strategy_key} on {symbol}")

            all_results[strategy_key] = strategy_results

            logger.info("=" * 60)
            logger.info(f"Summary for {strategy_key}")
            logger.info("=" * 60)

            if strategy_results:
                total_final_value = sum(r["final_value"] for r in strategy_results.values())
                avg_return = np.mean([r["total_return"] for r in strategy_results.values()])
                winning_tickers = sum(1 for r in strategy_results.values() if r["total_return"] > 0)

                logger.info(f"Tickers simulated: {len(strategy_results)}")
                logger.info(f"Total final value: ${total_final_value:,.2f}")
                logger.info(f"Average return: {avg_return * 100:.2f}%")
                logger.info(
                    f"Winning tickers: {winning_tickers}/{len(strategy_results)} ({winning_tickers / len(strategy_results) * 100:.1f}%)"
                )

        return self

    def rank_and_allocate(
        self,
        total_funds: float = 100_000.0,
        min_confidence: float = 0.6,
        max_allocation_pct: float = 1.0,
    ) -> pd.DataFrame:
        """Rank stocks by expected profitability and allocate capital.

        Uses the most recent prediction for each symbol to compute:
        - expected return
        - confidence score
        - historical volatility
        - priority score
        - proportional capital allocation

        :param total_funds: Total capital to allocate across all stocks.
        :param min_confidence: Minimum confidence score to include a stock (0–1).
        :param max_allocation_pct: Maximum fraction of *total_funds* for any single stock.
        :return: Ranked and allocated DataFrame.  Returns an empty DataFrame when
                 predictions are unavailable or no stock meets the threshold.
        """
        predictions = self._test_results.get("predictions") if self._test_results else None
        if predictions is None:
            print("⚠️  No predictions available. Run train() before rank_and_allocate().")
            return pd.DataFrame()

        pred_len = len(predictions)
        test_data: dict[str, Any] = {"prediction": predictions}
        test_data = self.validate_data_required_for_simulation(pred_len, test_data)
        test_df = pd.DataFrame(test_data)

        if test_df.empty or "symbol" not in test_df.columns:
            print("⚠️  Insufficient test data for ranking.")
            return pd.DataFrame()

        latest_per_symbol = (
            test_df.sort_values("date").groupby("symbol").last().reset_index()
        )

        pred_map = dict(zip(latest_per_symbol["symbol"], latest_per_symbol["prediction"]))
        price_map = dict(zip(latest_per_symbol["symbol"], latest_per_symbol["price"]))

        prices_history: dict[str, pd.Series] = {}
        for symbol in pred_map:
            sym_data = test_df[test_df["symbol"] == symbol].sort_values("date")
            if not sym_data.empty:
                prices_history[symbol] = sym_data.set_index("date")["price"]

        ranker = StockRanker(
            min_confidence=min_confidence,
            total_funds=total_funds,
            max_allocation_pct=max_allocation_pct,
        )

        ranked_df = ranker.rank_and_allocate(
            predictions=pred_map,
            current_prices=price_map,
            prices_history=prices_history,
            predictions_are_returns=True,
        )

        if ranked_df.empty:
            return ranked_df

        print("\n" + "=" * 60)
        print("Stock Ranking and Capital Allocation")
        print("=" * 60)
        print(f"Total funds: ${total_funds:,.2f}")
        print(f"Minimum confidence: {min_confidence:.0%}")
        print(f"Stocks ranked: {len(ranked_df)}\n")

        col_fmt = "{:<8} {:>14} {:>12} {:>12} {:>14} {:>5} {:>14}"
        header = col_fmt.format("Symbol", "Exp Return", "Confidence", "Volatility", "Priority", "Rank", "Allocation ($)")
        separator = "-" * len(header)
        print(header)
        print(separator)
        for _, row in ranked_df.iterrows():
            print(
                col_fmt.format(
                    row["symbol"],
                    f"{row['expected_return'] * 100:+.2f}%",
                    f"{row['confidence']:.4f}",
                    f"{row['volatility']:.6f}",
                    f"{row['priority_score']:.4f}",
                    int(row["rank"]),
                    f"${row['allocation']:,.2f}",
                )
            )
        print("=" * 60)

        return ranked_df

    def validate_data_required_for_simulation(self, pred_len: int, test_data: dict[str, Any]):
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

        return test_data

    def _optimize_hyperparameters(self, x_train_daily=None, y_train=None, x_test_daily=None, y_test=None):
        """
        Run optimization will only be run if the flag is enabled when the pipeline is instantiated.
        :param x_train_daily:
        :param y_train:
        :param x_test_daily:
        :param y_test:
        :return:
        """
        optimizer = StockModelOptimizer(x_train_daily, y_train, x_test_daily, y_test, n_trials=self.n_trials, n_jobs=1)
        optimizer.optimize_both()
        optimizer.visualize_studies(save_path="plots/optuna")
        optimizer.save_results(f"params/ALL_STOCKS_best_params.json")

        self._xgb_params, self._lgb_params = optimizer.best_xgb_params, optimizer.best_lgb_params

        return self
