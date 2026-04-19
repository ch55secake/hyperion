import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import yfinance as yf

from src.util import logger


class StockDataDownloader:
    """Downloads and manages stock data from yfinance"""

    # Stock info is cached to avoid downloading the same data multiple times
    _stock_info: dict[str, dict[str, Any]] = {}
    # History data is cached to avoid downloading the same data multiple times
    # Key is (symbol, period, interval)
    _history_data: dict[tuple[str, str, str], pd.DataFrame] = {}
    # Lock protecting writes to class-level shared dicts from multiple threads
    _lock: threading.Lock = threading.Lock()

    MAX_WORKERS: int = 16
    REQUIRED_COLUMNS: frozenset[str] = frozenset({"Open", "High", "Low", "Close", "Volume"})

    def __init__(self, symbols, period="2y", interval="1d"):
        self.symbols = symbols if isinstance(symbols, list) else [symbols]
        self.period = period
        self.interval = interval
        self.data = {}

        os.makedirs("./historic_data", exist_ok=True)

        if not self._stock_info:
            self._load_stock_info()

    @classmethod
    def _load_stock_info(cls, path="./historic_data/stock_info.json"):
        """
        Load cached stock info from disk if available.
        """
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="UTF-8") as f:
                    cls._stock_info = json.load(f)
                logger.info(f"Loaded cached stock info ({len(cls._stock_info)} entries)")
            except Exception as e:
                logger.warning(f"Failed to load cached stock info: {e}")

    @classmethod
    def save_stock_info(cls, path="./historic_data/stock_info.json"):
        """
        Save cached stock info to disk.
        """
        try:
            with open(path, "w", encoding="UTF-8") as f:
                json.dump(cls._stock_info, f, indent=4)
            logger.info(f"Saved stock info ({len(cls._stock_info)} entries)")
        except Exception as e:
            logger.warning(f"Failed to save stock info: {e}")

    def _download_single(self, symbol: str) -> tuple[str, pd.DataFrame | None]:
        """
        Download or load cached data for a single symbol.
        Exceptions are allowed to propagate so the caller can track failures.
        :param symbol: the stock symbol to download
        :return: tuple of (symbol, DataFrame) — DataFrame is None for empty data
        """
        default_path: str = "./historic_data/"
        filename = f"{symbol}_{self.period}_{self.interval}.parquet"
        complete_path: str = os.path.join(default_path, filename)
        logger.debug(f"Checking for existing data for {complete_path}...")

        needs_refresh = False

        if os.path.isfile(complete_path):
            df: pd.DataFrame | None = None
            try:
                df = pd.read_parquet(complete_path)
            except Exception as e:
                logger.warning(f"Failed to read cached parquet for {symbol}: {e}. Re-downloading.")
                needs_refresh = True

            if df is not None:
                missing_cols = self.REQUIRED_COLUMNS - set(df.columns)
                if missing_cols:
                    logger.warning(f"Cached parquet for {symbol} is missing columns {missing_cols}. Re-downloading.")
                    needs_refresh = True
                else:
                    last_date = pd.to_datetime(df.index[-1]).date()
                    lookback_date = (datetime.now() - timedelta(days=2)).date()

                    if last_date < lookback_date:
                        logger.warning(f"Cache is outdated (last date: {last_date}, lookback date: {lookback_date})")
                        needs_refresh = True
                    else:
                        logger.info(f"Using cached data for {symbol}")
                        with self._lock:
                            self._history_data[(symbol, self.period, self.interval)] = df
                        return symbol, df

        if not os.path.isfile(complete_path) or needs_refresh:
            logger.info(f"Downloading {symbol} ({self.period} {self.interval} data)...")
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=self.period, interval=self.interval)

            with self._lock:
                self._history_data[(symbol, self.period, self.interval)] = df

            if df.empty:
                logger.warning(f"No data found for {symbol}")
                return symbol, None

            parquet_path = f"./historic_data/{filename}"
            df.to_parquet(parquet_path)

            with self._lock:
                self._stock_info[symbol] = ticker.info

            logger.info(f"Downloaded {len(df)} data points for {symbol}")
            logger.info(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
            logger.info(f"Saved to {parquet_path}")

            return symbol, df

        return symbol, None

    def download_data(self) -> tuple[dict, list[str]]:
        """Download data for all symbols in parallel using a thread pool.

        Returns:
            A tuple of (data, failed) where data maps symbol -> DataFrame for
            successfully downloaded symbols, and failed is a list of symbols
            that could not be downloaded.
        """
        logger.info("=" * 60)
        logger.info("Downloading Stock Data from yfinance")
        logger.info("=" * 60)

        failed: list[str] = []

        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            futures = {executor.submit(self._download_single, sym): sym for sym in self.symbols}
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    _, df = future.result()
                    if df is not None:
                        self.data[symbol] = df
                except Exception as e:
                    logger.error(f"Error downloading {symbol}: {str(e)}")
                    failed.append(symbol)

        self.save_stock_info()

        if failed:
            logger.warning("%d symbol(s) failed to download: %s", len(failed), failed)

        return self.data, failed

    @staticmethod
    def _get_stock_info(symbol) -> None:
        """
        Get stock info from yfinance
        :param symbol: the stock symbol
        """
        if symbol not in StockDataDownloader._stock_info.keys():
            StockDataDownloader._stock_info[symbol] = yf.Ticker(symbol).info

    @staticmethod
    def get_sector(symbol):
        """
        Get the sector of a given stock
        :param symbol: the stock you want the sector for
        :return: sector or unknown if it is not found
        """
        StockDataDownloader._get_stock_info(symbol)
        return StockDataDownloader._stock_info[symbol].get("sector", "Unknown")

    @staticmethod
    def get_market_cap(symbol):
        """
        Get the market cap of a given stock
        :param symbol:
        :return:
        """
        StockDataDownloader._get_stock_info(symbol)
        data = StockDataDownloader._stock_info[symbol]

        market_cap = data.get("marketCap", None)

        if market_cap is None or market_cap == 0:
            logger.warning(f"Failed to get market cap for {symbol}")
            shares = data.get("sharesOutstanding", None)
            price = data.get("currentPrice", None)
            if shares and price:
                market_cap = shares * price
            else:
                logger.warning(f"Unable to compute market cap for {symbol}, returning None")
                return None

        return market_cap

    @staticmethod
    def get_industry(symbol):
        """
        Get the industry of a given stock
        :param symbol: the stock you want the industry for
        :return: the industry or unknown if it is not found
        """
        StockDataDownloader._get_stock_info(symbol)
        return StockDataDownloader._stock_info[symbol].get("industry", "Unknown")

    @staticmethod
    def get_beta(symbol):
        """
        Get the beta of a given stock (volatility vs market)
        :param symbol: the stock you want the beta for
        :return: beta or 1.0 if it is not found
        """
        StockDataDownloader._get_stock_info(symbol)
        return StockDataDownloader._stock_info[symbol].get("beta", 1.0)

    def get_avg_volume(self, symbol):
        """
        Get the average volume of a given stock
        :param symbol: the stock you want the average volume for
        :return: average volume
        """
        if (symbol, self.period, self.interval) not in self._history_data:
            self._history_data[(symbol, self.period, self.interval)] = yf.Ticker(symbol).history(
                period=self.period, interval=self.interval
            )
        return self._history_data[(symbol, self.period, self.interval)]["Volume"].mean()
