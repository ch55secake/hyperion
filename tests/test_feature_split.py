"""Unit tests for src/feature/feature_split.py"""

from src.feature.feature_split import derive_feature_split, DEFAULT_SHARED_FEATURES


class TestDeriveFeatureSplit:
    def test_shared_features_go_to_shared(self):
        columns = list(DEFAULT_SHARED_FEATURES)
        result = derive_feature_split(columns)
        for col in DEFAULT_SHARED_FEATURES:
            assert col in result["shared"]
        assert result["daily"] == []
        assert result["hourly"] == []

    def test_large_window_goes_to_daily(self):
        # A column with a window >= 20 should appear in daily
        result = derive_feature_split(["SMA_20"])
        assert "SMA_20" in result["daily"]

    def test_small_window_goes_to_hourly(self):
        # A column with a window < 20 only should appear in hourly only
        result = derive_feature_split(["EMA_5"])
        assert "EMA_5" in result["hourly"]
        assert "EMA_5" not in result["daily"]

    def test_column_in_both_daily_and_hourly(self):
        # A column with windows that cross the threshold (e.g. 5 and 50)
        result = derive_feature_split(["Ratio_5_50"])
        assert "Ratio_5_50" in result["daily"]
        assert "Ratio_5_50" in result["hourly"]

    def test_no_window_column_goes_to_shared(self):
        result = derive_feature_split(["Open", "Close"])
        assert "Open" in result["shared"]
        assert "Close" in result["shared"]

    def test_per_column_keys_match_input(self):
        columns = ["SMA_20", "EMA_5", "ticker"]
        result = derive_feature_split(columns)
        assert set(result["per_column"].keys()) == set(columns)

    def test_per_column_values_are_sorted(self):
        columns = ["Ratio_5_50", "ticker"]
        result = derive_feature_split(columns)
        for buckets in result["per_column"].values():
            assert buckets == sorted(buckets)

    def test_empty_columns(self):
        result = derive_feature_split([])
        assert result["daily"] == []
        assert result["hourly"] == []
        assert result["shared"] == []
        assert result["per_column"] == {}

    def test_custom_threshold(self):
        # With threshold=10, a window of 10 should go to daily
        result = derive_feature_split(["SMA_10"], short_term_threshold=10)
        assert "SMA_10" in result["daily"]
        # Window of 5 should go to hourly (< 10)
        result2 = derive_feature_split(["EMA_5"], short_term_threshold=10)
        assert "EMA_5" in result2["hourly"]

    def test_duplicate_columns_handled(self):
        # Providing the same column twice should not cause errors
        result = derive_feature_split(["SMA_20", "SMA_20"])
        assert isinstance(result["daily"], list)
