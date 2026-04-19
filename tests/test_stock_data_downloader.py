"""Unit tests for src/data/stock_data_downloader.py"""

import os
import tempfile
import threading
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.stock_data_downloader import StockDataDownloader


def _make_price_df(n=10):
    """Create a minimal price DataFrame with a fresh date index."""
    today = pd.Timestamp.now().normalize()
    dates = pd.date_range(end=today, periods=n, freq="D")
    return pd.DataFrame(
        {
            "Open": [100.0] * n,
            "High": [105.0] * n,
            "Low": [95.0] * n,
            "Close": [102.0] * n,
            "Volume": [1_000_000] * n,
        },
        index=dates,
    )


@pytest.fixture(autouse=True)
def reset_class_state(tmp_path, monkeypatch):
    """Isolate class-level state and redirect file I/O to a temp directory."""
    StockDataDownloader._stock_info = {}
    StockDataDownloader._history_data = {}
    monkeypatch.chdir(tmp_path)
    os.makedirs(tmp_path / "historic_data", exist_ok=True)
    yield
    StockDataDownloader._stock_info = {}
    StockDataDownloader._history_data = {}


class TestDownloadDataParallel:
    """Tests verifying the parallel download path works correctly."""

    def test_returns_data_for_all_symbols(self):
        df = _make_price_df()
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = df
        mock_ticker.info = {"sector": "Tech", "industry": "Software"}

        with patch("src.data.stock_data_downloader.yf.Ticker", return_value=mock_ticker):
            downloader = StockDataDownloader(["AAPL", "MSFT", "GOOG"], period="1mo", interval="1d")
            result = downloader.download_data()

        assert set(result.keys()) == {"AAPL", "MSFT", "GOOG"}

    def test_empty_data_excluded_from_result(self):
        good_df = _make_price_df()
        empty_df = pd.DataFrame()

        def _ticker_factory(symbol):
            mock = MagicMock()
            mock.history.return_value = empty_df if symbol == "BAD" else good_df
            mock.info = {}
            return mock

        with patch("src.data.stock_data_downloader.yf.Ticker", side_effect=_ticker_factory):
            downloader = StockDataDownloader(["AAPL", "BAD"], period="1mo", interval="1d")
            result = downloader.download_data()

        assert "AAPL" in result
        assert "BAD" not in result

    def test_error_in_one_symbol_does_not_block_others(self):
        good_df = _make_price_df()

        def _ticker_factory(symbol):
            mock = MagicMock()
            if symbol == "ERR":
                mock.history.side_effect = RuntimeError("network error")
            else:
                mock.history.return_value = good_df
                mock.info = {}
            return mock

        with patch("src.data.stock_data_downloader.yf.Ticker", side_effect=_ticker_factory):
            downloader = StockDataDownloader(["AAPL", "ERR", "MSFT"], period="1mo", interval="1d")
            result = downloader.download_data()

        assert "AAPL" in result
        assert "MSFT" in result
        assert "ERR" not in result

    def test_csv_written_for_downloaded_symbols(self, tmp_path):
        df = _make_price_df()
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = df
        mock_ticker.info = {}

        with patch("src.data.stock_data_downloader.yf.Ticker", return_value=mock_ticker):
            downloader = StockDataDownloader(["AAPL"], period="1mo", interval="1d")
            downloader.download_data()

        assert os.path.isfile("./historic_data/AAPL_1mo_1d.csv")

    def test_cached_csv_used_without_network_call(self, tmp_path):
        df = _make_price_df()
        df.to_csv("./historic_data/AAPL_1mo_1d.csv")

        with patch("src.data.stock_data_downloader.yf.Ticker") as mock_yf:
            downloader = StockDataDownloader(["AAPL"], period="1mo", interval="1d")
            result = downloader.download_data()
            mock_yf.assert_not_called()

        assert "AAPL" in result

    def test_history_data_cache_populated(self):
        df = _make_price_df()
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = df
        mock_ticker.info = {}

        with patch("src.data.stock_data_downloader.yf.Ticker", return_value=mock_ticker):
            downloader = StockDataDownloader(["AAPL"], period="1mo", interval="1d")
            downloader.download_data()

        assert ("AAPL", "1mo", "1d") in StockDataDownloader._history_data

    def test_class_level_lock_exists(self):
        assert isinstance(StockDataDownloader._lock, type(threading.Lock()))

    def test_max_workers_constant_defined(self):
        assert isinstance(StockDataDownloader.MAX_WORKERS, int)
        assert StockDataDownloader.MAX_WORKERS > 0

    def test_no_info_fetch_for_cached_fresh_data(self, tmp_path):
        """Redundant .info call must NOT be made when data is loaded from a fresh cache."""
        df = _make_price_df()
        df.to_csv("./historic_data/TSLA_1mo_1d.csv")

        with patch("src.data.stock_data_downloader.yf.Ticker") as mock_yf:
            downloader = StockDataDownloader(["TSLA"], period="1mo", interval="1d")
            downloader.download_data()
            mock_yf.assert_not_called()
