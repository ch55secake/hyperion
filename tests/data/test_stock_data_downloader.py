"""Unit tests for src/data/stock_data_downloader.py (StockDataDownloader)."""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

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


def _make_fresh_price_df(n=10):
    """Create a price DataFrame with a date index ending today (cache-fresh)."""
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


def _make_tz_aware_df(tz="America/New_York"):
    """Return a small DataFrame with a timezone-aware DatetimeIndex, mimicking yfinance output."""
    index = pd.date_range("2024-01-01", periods=5, freq="D", tz=tz)
    return pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "Close": [101.0, 102.0, 103.0, 104.0, 105.0],
            "Volume": [1000, 1100, 1200, 1300, 1400],
        },
        index=index,
    )


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


# ---------------------------------------------------------------------------
# Parallel download behaviour
# ---------------------------------------------------------------------------


class TestDownloadDataParallel:
    """Tests verifying the parallel download path works correctly."""

    def setup_method(self):
        _reset_class_caches()

    def test_returns_data_for_all_symbols(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        os.makedirs(tmp_path / "historic_data", exist_ok=True)
        df = _make_fresh_price_df()
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = df
        mock_ticker.info = {"sector": "Tech", "industry": "Software"}

        with patch("src.data.stock_data_downloader.yf.Ticker", return_value=mock_ticker):
            downloader = StockDataDownloader(["AAPL", "MSFT", "GOOG"], period="1mo", interval="1d")
            result, failed = downloader.download_data()

        assert set(result.keys()) == {"AAPL", "MSFT", "GOOG"}
        assert failed == []

    def test_empty_data_excluded_from_result(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        os.makedirs(tmp_path / "historic_data", exist_ok=True)
        good_df = _make_fresh_price_df()
        empty_df = pd.DataFrame()

        def _ticker_factory(symbol):
            mock = MagicMock()
            mock.history.return_value = empty_df if symbol == "BAD" else good_df
            mock.info = {}
            return mock

        with patch("src.data.stock_data_downloader.yf.Ticker", side_effect=_ticker_factory):
            downloader = StockDataDownloader(["AAPL", "BAD"], period="1mo", interval="1d")
            result, failed = downloader.download_data()

        assert "AAPL" in result
        assert "BAD" not in result
        # Empty data is skipped but not counted as a failure
        assert "BAD" not in failed

    def test_error_in_one_symbol_tracked_in_failed(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        os.makedirs(tmp_path / "historic_data", exist_ok=True)
        good_df = _make_fresh_price_df()

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
            result, failed = downloader.download_data()

        assert "AAPL" in result
        assert "MSFT" in result
        assert "ERR" not in result
        assert "ERR" in failed

    def test_parquet_written_for_downloaded_symbols(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        os.makedirs(tmp_path / "historic_data", exist_ok=True)
        df = _make_fresh_price_df()
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = df
        mock_ticker.info = {}

        with patch("src.data.stock_data_downloader.yf.Ticker", return_value=mock_ticker):
            downloader = StockDataDownloader(["AAPL"], period="1mo", interval="1d")
            downloader.download_data()

        assert os.path.isfile("./historic_data/AAPL_1mo_1d.parquet")

    def test_cached_parquet_used_without_network_call(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        os.makedirs(tmp_path / "historic_data", exist_ok=True)
        df = _make_fresh_price_df()
        df.to_parquet("./historic_data/AAPL_1mo_1d.parquet")

        with patch("src.data.stock_data_downloader.yf.Ticker") as mock_yf:
            downloader = StockDataDownloader(["AAPL"], period="1mo", interval="1d")
            result, _ = downloader.download_data()
            mock_yf.assert_not_called()

        assert "AAPL" in result

    def test_history_data_cache_populated(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        os.makedirs(tmp_path / "historic_data", exist_ok=True)
        df = _make_fresh_price_df()
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = df
        mock_ticker.info = {}

        with patch("src.data.stock_data_downloader.yf.Ticker", return_value=mock_ticker):
            downloader = StockDataDownloader(["AAPL"], period="1mo", interval="1d")
            downloader.download_data()

        assert ("AAPL", "1mo", "1d") in StockDataDownloader._history_data

    def test_class_level_lock_exists(self):
        assert hasattr(StockDataDownloader._lock, "acquire")
        assert hasattr(StockDataDownloader._lock, "release")

    def test_max_workers_constant_defined(self):
        assert isinstance(StockDataDownloader.MAX_WORKERS, int)
        assert StockDataDownloader.MAX_WORKERS > 0

    def test_no_info_fetch_for_cached_fresh_data(self, tmp_path, monkeypatch):
        """Redundant .info call must NOT be made when data is loaded from a fresh cache."""
        monkeypatch.chdir(tmp_path)
        os.makedirs(tmp_path / "historic_data", exist_ok=True)
        df = _make_fresh_price_df()
        df.to_parquet("./historic_data/TSLA_1mo_1d.parquet")

        with patch("src.data.stock_data_downloader.yf.Ticker") as mock_yf:
            downloader = StockDataDownloader(["TSLA"], period="1mo", interval="1d")
            downloader.download_data()
            mock_yf.assert_not_called()


# ---------------------------------------------------------------------------
# Parquet round-trip: timezone preservation
# ---------------------------------------------------------------------------


class TestParquetRoundTrip:
    """Verify that the Parquet cache preserves timezone-aware DatetimeIndex."""

    def test_timezone_preserved_after_roundtrip(self, tmp_path):
        df = _make_tz_aware_df()
        path = tmp_path / "test.parquet"

        df.to_parquet(path)
        df_loaded = pd.read_parquet(path)

        assert df_loaded.index.tzinfo is not None, "Timezone info should be preserved after Parquet round-trip"
        assert str(df_loaded.index.tz) == str(df.index.tz), "Timezone should match original after round-trip"

    def test_index_values_preserved_after_roundtrip(self, tmp_path):
        df = _make_tz_aware_df()
        path = tmp_path / "test.parquet"

        df.to_parquet(path)
        df_loaded = pd.read_parquet(path)

        # Parquet does not store DatetimeIndex frequency; compare without freq
        pd.testing.assert_index_equal(df.index, df_loaded.index, exact=False, check_names=True)

    def test_data_values_preserved_after_roundtrip(self, tmp_path):
        df = _make_tz_aware_df()
        path = tmp_path / "test.parquet"

        df.to_parquet(path)
        df_loaded = pd.read_parquet(path)

        pd.testing.assert_frame_equal(df, df_loaded, check_freq=False)


# ---------------------------------------------------------------------------
# Failure-path tests
# ---------------------------------------------------------------------------


class TestDownloadFailurePaths:
    """Tests for failure scenarios in download_data()."""

    def setup_method(self):
        _reset_class_caches()

    @pytest.mark.parametrize("fail_count,total", [(1, 3), (2, 4), (3, 3)])
    def test_partial_failure_returns_successful_data(self, tmp_path, monkeypatch, fail_count, total):
        """N of M tickers fail — the other M-N are returned and failures are tracked."""
        monkeypatch.chdir(tmp_path)
        os.makedirs(tmp_path / "historic_data", exist_ok=True)

        symbols = [f"SYM{i}" for i in range(total)]
        failing = set(symbols[:fail_count])
        good_df = _make_fresh_price_df()

        def _ticker_factory(symbol):
            mock = MagicMock()
            if symbol in failing:
                mock.history.side_effect = RuntimeError("network error")
            else:
                mock.history.return_value = good_df
                mock.info = {}
            return mock

        with patch("src.data.stock_data_downloader.yf.Ticker", side_effect=_ticker_factory):
            downloader = StockDataDownloader(symbols, period="1mo", interval="1d")
            result, failed = downloader.download_data()

        assert set(result.keys()) == set(symbols) - failing
        assert set(failed) == failing

    def test_all_tickers_fail_returns_empty_dict(self, tmp_path, monkeypatch):
        """When all tickers fail the result must be an empty dict — no crash."""
        monkeypatch.chdir(tmp_path)
        os.makedirs(tmp_path / "historic_data", exist_ok=True)

        def _ticker_factory(symbol):
            mock = MagicMock()
            mock.history.side_effect = RuntimeError("network unreachable")
            return mock

        with patch("src.data.stock_data_downloader.yf.Ticker", side_effect=_ticker_factory):
            downloader = StockDataDownloader(["A", "B", "C"], period="1mo", interval="1d")
            result, failed = downloader.download_data()

        assert result == {}
        assert set(failed) == {"A", "B", "C"}

    def test_corrupted_parquet_triggers_redownload(self, tmp_path, monkeypatch):
        """A truncated / invalid parquet file should trigger a re-download, not a crash."""
        monkeypatch.chdir(tmp_path)
        historic_dir = tmp_path / "historic_data"
        os.makedirs(historic_dir, exist_ok=True)

        # Write garbage bytes to simulate a corrupted parquet
        corrupt_path = historic_dir / "AAPL_1mo_1d.parquet"
        corrupt_path.write_bytes(b"this is not a valid parquet file")

        good_df = _make_fresh_price_df()
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = good_df
        mock_ticker.info = {}

        with patch("src.data.stock_data_downloader.yf.Ticker", return_value=mock_ticker) as mock_yf:
            downloader = StockDataDownloader(["AAPL"], period="1mo", interval="1d")
            result, failed = downloader.download_data()
            mock_yf.assert_called_once_with("AAPL")

        assert "AAPL" in result
        assert failed == []

    def test_schema_drift_triggers_redownload(self, tmp_path, monkeypatch):
        """A cached parquet missing required columns must trigger a re-download."""
        monkeypatch.chdir(tmp_path)
        os.makedirs(tmp_path / "historic_data", exist_ok=True)

        # Write a parquet that is missing 'Close' and 'Volume' (schema drift)
        stale_df = _make_fresh_price_df().drop(columns=["Close", "Volume"])
        stale_df.to_parquet("./historic_data/AAPL_1mo_1d.parquet")

        good_df = _make_fresh_price_df()
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = good_df
        mock_ticker.info = {}

        with patch("src.data.stock_data_downloader.yf.Ticker", return_value=mock_ticker) as mock_yf:
            downloader = StockDataDownloader(["AAPL"], period="1mo", interval="1d")
            result, failed = downloader.download_data()
            mock_yf.assert_called_once_with("AAPL")

        assert "AAPL" in result
        assert "Close" in result["AAPL"].columns
        assert "Volume" in result["AAPL"].columns
        assert failed == []
