import yfinance as yf

class StockDataDownloader:
    """Downloads and manages stock data from yfinance"""

    def __init__(self, symbols, period='2y', interval='1d'):
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
                print(f"\nDownloading {symbol}...")
                print(f"\n{self.period} {self.interval} data for {symbol}")
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=self.period, interval=self.interval)

                if df.empty:
                    print(f"  ⚠️  No data found for {symbol}")
                    continue

                # Save to CSV
                filename = f"./historic_data/{symbol}_{self.period}_{self.interval}.csv"
                df.to_csv(filename)

                self.data[symbol] = df
                print(f"  ✓ Downloaded {len(df)} data points")
                print(f"  ✓ Date range: {df.index[0].date()} to {df.index[-1].date()}")
                print(f"  ✓ Saved to {filename}")

            except Exception as e:
                print(f"  ✗ Error downloading {symbol}: {str(e)}")

        return self.data