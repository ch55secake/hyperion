"""Unit tests for src/data/stock_data_downloader.py (StockDataDownloader)."""

import json
import os
import tempfile

import pandas as pd
import pytest

from src.data.stock_data_downloader import StockDataDownloader

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reset_class_caches():
    """Clear class-level caches so tests do not interfere with each other."""
    StockDataDownloader._stock_info = {}
    StockDataDownloader._history_data = {}


def _inject_stock_info(symbol: str, info: dict):
    StockDataDownloader._stock_info[symbol] = info


def _make_price_df(n=50, symbol="FAKE"):
    idx = pd.date_range("2022-01-01", periods=n, freq="D")
    df = pd.DataFrame(
        {
            "Open": 100.0,
            "High": 105.0,
            "Low": 95.0,
            "Close": 102.0,
            "Volume": 1_000_000.0,
        },
        index=idx,
    )
    return df


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestStockDataDownloaderInit:
    def setup_method(self):
        _reset_class_caches()

    def test_single_symbol_wrapped_in_list(self):
        dl = StockDataDownloader("AAPL", period="1y", interval="1d")
        assert dl.symbols == ["AAPL"]

    def test_list_of_symbols_stored(self):
        dl = StockDataDownloader(["AAPL", "MSFT"], period="1y", interval="1d")
        assert dl.symbols == ["AAPL", "MSFT"]

    def test_period_and_interval_stored(self):
        dl = StockDataDownloader(["AAPL"], period="6mo", interval="1h")
        assert dl.period == "6mo"
        assert dl.interval == "1h"

    def test_data_starts_empty(self):
        dl = StockDataDownloader(["AAPL"])
        assert dl.data == {}


# ---------------------------------------------------------------------------
# save_stock_info / _load_stock_info roundtrip
# ---------------------------------------------------------------------------


class TestSaveLoadStockInfo:
    def setup_method(self):
        _reset_class_caches()

    def test_save_creates_file(self):
        _inject_stock_info("FAKE", {"sector": "Tech"})
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "stock_info.json")
            StockDataDownloader.save_stock_info(path=path)
            assert os.path.isfile(path)

    def test_load_restores_info(self):
        _inject_stock_info("FAKE", {"sector": "Tech", "industry": "Software"})
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "stock_info.json")
            StockDataDownloader.save_stock_info(path=path)
            # Clear and reload
            _reset_class_caches()
            StockDataDownloader._load_stock_info(path=path)
            assert StockDataDownloader._stock_info.get("FAKE", {}).get("sector") == "Tech"

    def test_load_non_existent_file_does_not_raise(self):
        # A missing file should be silently skipped
        StockDataDownloader._load_stock_info(path="/tmp/no_such_file_xyz.json")

    def test_save_writes_valid_json(self):
        _inject_stock_info("FAKE", {"beta": 1.2})
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "stock_info.json")
            StockDataDownloader.save_stock_info(path=path)
            with open(path) as f:
                data = json.load(f)
            assert data["FAKE"]["beta"] == 1.2


# ---------------------------------------------------------------------------
# get_sector / get_industry / get_beta
# ---------------------------------------------------------------------------


class TestGetStockMetadata:
    def setup_method(self):
        _reset_class_caches()
        _inject_stock_info(
            "FAKE",
            {
                "sector": "Technology",
                "industry": "Software",
                "beta": 1.5,
                "marketCap": 1_000_000,
            },
        )

    def test_get_sector_returns_correct_value(self):
        sector = StockDataDownloader.get_sector("FAKE")
        assert sector == "Technology"

    def test_get_sector_missing_returns_unknown(self):
        _inject_stock_info("NOSEC", {})
        assert StockDataDownloader.get_sector("NOSEC") == "Unknown"

    def test_get_industry_returns_correct_value(self):
        assert StockDataDownloader.get_industry("FAKE") == "Software"

    def test_get_industry_missing_returns_unknown(self):
        _inject_stock_info("NOIND", {})
        assert StockDataDownloader.get_industry("NOIND") == "Unknown"

    def test_get_beta_returns_correct_value(self):
        assert StockDataDownloader.get_beta("FAKE") == pytest.approx(1.5)

    def test_get_beta_missing_returns_default_one(self):
        _inject_stock_info("NOBETA", {})
        assert StockDataDownloader.get_beta("NOBETA") == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# get_market_cap
# ---------------------------------------------------------------------------


class TestGetMarketCap:
    def setup_method(self):
        _reset_class_caches()

    def test_returns_market_cap_when_present(self):
        _inject_stock_info("FAKE", {"marketCap": 5_000_000})
        result = StockDataDownloader.get_market_cap("FAKE")
        assert result == 5_000_000

    def test_falls_back_to_shares_times_price(self):
        _inject_stock_info(
            "FAKE2",
            {
                "marketCap": 0,
                "sharesOutstanding": 1_000,
                "currentPrice": 100.0,
            },
        )
        result = StockDataDownloader.get_market_cap("FAKE2")
        assert result == pytest.approx(100_000.0)

    def test_returns_none_when_no_fallback_data(self):
        _inject_stock_info("FAKE3", {"marketCap": 0})
        result = StockDataDownloader.get_market_cap("FAKE3")
        assert result is None


# ---------------------------------------------------------------------------
# get_avg_volume
# ---------------------------------------------------------------------------


class TestGetAvgVolume:
    def setup_method(self):
        _reset_class_caches()

    def test_returns_mean_volume_from_cached_data(self):
        symbol = "CACHE"
        df = _make_price_df(n=30)
        StockDataDownloader._history_data[(symbol, "1y", "1d")] = df
        dl = StockDataDownloader(symbol, period="1y", interval="1d")
        avg = dl.get_avg_volume(symbol)
        assert avg == pytest.approx(1_000_000.0)
