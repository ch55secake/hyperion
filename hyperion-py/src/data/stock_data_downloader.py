import logging
import os
from typing import Any

import pandas as pd
import yfinance as yf

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

    def download_data(self):
        """Download data for all symbols"""
        print("=" * 60)
        print("Downloading Stock Data from yfinance")
        print("=" * 60)

        for symbol in self.symbols:
            try:
                # TODO: Double check this slop
                default_path: str = "./historic_data/"
                filename: str = symbol + "_2y_1h" + ".csv"
                complete_path: str = os.path.join(default_path, filename)
                print(f"\nChecking for existing data for {complete_path}...")
                if os.path.isfile(complete_path):
                    self._history_data[(symbol, self.period, self.interval)] = pd.read_csv(complete_path)
                if (
                    symbol in self.data
                    and (symbol, self.period, self.interval) in self._history_data.keys()
                    and symbol in self._stock_info.keys()
                ):
                    # No check on if data is in file, technically shouldn't be needed
                    print(f"\nData already downloaded for {symbol}")
                    continue
                print(f"\nDownloading {symbol}...")
                print(f"\n{self.period} {self.interval} data for {symbol}")
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=self.period, interval=self.interval)

                self._history_data[(symbol, self.period, self.interval)] = df
                df = df.resample("1D").last()

                if df.empty:
                    print(f"  ⚠️  No data found for {symbol}")
                    continue

                filename = f"./historic_data/{symbol}_{self.period}_{self.interval}.csv"
                df.to_csv(filename)

                self.data[symbol] = df
                self._stock_info[symbol] = ticker.info
                print(f"  ✓ Downloaded {len(df)} data points")
                print(f"  ✓ Date range: {df.index[0].date()} to {df.index[-1].date()}")
                print(f"  ✓ Saved to {filename}")

            except Exception as e:
                print(f"  ✗ Error downloading {symbol}: {str(e)}")

        return self.data

    @staticmethod
    def get_sector(symbol):
        """
        Get the sector of a given stock
        :param symbol: the stock you want the sector for
        :return: sector or unknown if it is not found
        """
        return StockDataDownloader._stock_info.get(symbol, yf.Ticker(symbol)).get("sector", "Unknown")

    @staticmethod
    def get_market_cap(symbol):
        """
        Get the market cap of a given stock
        :param symbol:
        :return:
        """

        data = StockDataDownloader._stock_info.get(symbol, yf.Ticker(symbol).info)

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
        return StockDataDownloader._stock_info.get(symbol, yf.Ticker(symbol)).get("industry", "Unknown")

    @staticmethod
    def get_beta(symbol):
        """
        Get the beta of a given stock (volatility vs market)
        :param symbol: the stock you want the beta for
        :return: beta or 1.0 if it is not found
        """
        return StockDataDownloader._stock_info.get(symbol, yf.Ticker(symbol)).get("beta", 1.0)

    def get_avg_volume(self, symbol):
        """
        Get the average volume of a given stock
        :param symbol: the stock you want the average volume for
        :return: average volume
        """
        return self._history_data.get(
            (symbol, self.period, self.interval), yf.Ticker(symbol).history(period=self.period, interval=self.interval)
        )["Volume"].mean()
