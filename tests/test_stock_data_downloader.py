"""Unit tests for src/data/stock_data_downloader.py"""

import os
import tempfile

import pandas as pd
import pytest


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
