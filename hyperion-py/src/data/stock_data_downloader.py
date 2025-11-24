import json
import logging
import os
from typing import Any

import pandas as pd
import yfinance as yf

from src.console import ConsoleFormatter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockDataDownloader:
    """Downloads and manages stock data from yfinance"""

    # Stock info is cached to avoid downloading the same data multiple times
    _stock_info: dict[str, dict[str, Any]] = {}
    # History data is cached to avoid downloading the same data multiple times
    # Key is (symbol, period, interval)
    _history_data: dict[tuple[str, str, str], pd.DataFrame] = {}

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
        """Load cached stock info from disk if available."""

        if os.path.isfile(path):
            try:
                with open(path, "r") as f:
                    cls._stock_info = json.load(f)
                ConsoleFormatter.success(f"Loaded cached stock info ({len(cls._stock_info)} entries)")
            except Exception as e:
                ConsoleFormatter.warning(f"Failed to load cached stock info: {e}")

    @classmethod
    def save_stock_info(cls, path="./historic_data/stock_info.json"):
        """Save cached stock info to disk."""

        try:
            with open(path, "w") as f:
                json.dump(cls._stock_info, f, indent=4)
            ConsoleFormatter.success(f"Saved stock info ({len(cls._stock_info)} entries)", new_lines_before_message=1)
        except Exception as e:
            ConsoleFormatter.warning(f"Failed to save stock info: {e}", new_lines_before_message=1)


    def download_data(self):
        """Download data for all symbols"""
        ConsoleFormatter.new_section("Downloading Stock Data from yfinance", new_lines_before_message=1)

        for symbol in self.symbols:
            try:
                default_path: str = "./historic_data/"
                filename = f"{symbol}_{self.period}_{self.interval}.csv"
                complete_path: str = os.path.join(default_path, filename)
                ConsoleFormatter.info(f"Checking for existing data for {complete_path}...", new_lines_before_message=1)

                if os.path.isfile(complete_path):
                    ConsoleFormatter.success(f"Using cached data for {symbol}", indentation=1)
                    df = pd.read_csv(complete_path, parse_dates=True, index_col=0)

                    self._history_data[(symbol, self.period, self.interval)] = df
                    self.data[symbol] = df
                    if symbol not in self._stock_info:
                        self._stock_info[symbol] = yf.Ticker(symbol).info
                    continue

                ConsoleFormatter.info(f"Downloading {symbol}...", new_lines_before_message=1)
                ConsoleFormatter.info(f"{self.period} {self.interval} data for {symbol}", new_lines_before_message=1)
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=self.period, interval=self.interval)

                self._history_data[(symbol, self.period, self.interval)] = df
                df = df.resample("1D").last()

                if df.empty:
                    ConsoleFormatter.error(f"No data found for {symbol}", indentation=1)
                    continue

                filename = f"./historic_data/{filename}"
                df.to_csv(filename)

                self.data[symbol] = df
                self._stock_info[symbol] = ticker.info

                ConsoleFormatter.success(f"Downloaded {len(df)} data points", indentation=1)
                ConsoleFormatter.success(f"Date range: {df.index[0].date()} to {df.index[-1].date()}", indentation=1)
                ConsoleFormatter.success(f"Saved to {filename}", indentation=1)

            except Exception as e:
                ConsoleFormatter.error(f"Error downloading {symbol}: {str(e)}", indentation=1)

        self.save_stock_info()
        return self.data

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
            logger.warning("Failed to get market cap for " + symbol)
            shares = data.get("sharesOutstanding", None)
            price = data.get("currentPrice", None)
            if shares and price:
                market_cap = shares * price
            else:
                # TODO: Also be cautious of this, defaulting to a billion will be slightly dangerous, might lead to
                # some ropey preds
                return 1e9  # Default to 1 billion

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
            self._history_data[(symbol, self.period, self.interval)] = yf.Ticker(symbol).history(period=self.period, interval=self.interval)
        return self._history_data[(symbol, self.period, self.interval)]["Volume"].mean()
