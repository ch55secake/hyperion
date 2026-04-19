"""Unit tests for src/align/align.py"""

import numpy as np
import pandas as pd
import pytest

from src.align import align_targets_across_intervals, ensure_prediction_alignment
from src.align.align import align_to_reference


def _daily_series(n, start="2022-01-01"):
    idx = pd.date_range(start, periods=n, freq="D")
    return pd.Series(np.arange(float(n)), index=idx)


class TestAlignToReference:
    def test_ffill_same_length_preserves_values(self):
        series = _daily_series(10)
        ref = pd.date_range("2022-01-01", periods=10, freq="D")
        result = align_to_reference(series, ref, method="ffill")
        assert len(result) == 10
        assert list(result.index) == list(ref)

    def test_ffill_series_shorter_than_ref_truncates_to_min(self):
        series = _daily_series(5)
        ref = pd.date_range("2022-01-01", periods=10, freq="D")
        result = align_to_reference(series, ref, method="ffill")
        assert len(result) == 5

    def test_ffill_series_longer_than_ref_truncates_to_ref(self):
        series = _daily_series(20)
        ref = pd.date_range("2022-01-01", periods=10, freq="D")
        result = align_to_reference(series, ref, method="ffill")
        assert len(result) == 10

    def test_ffill_values_are_original_values(self):
        series = _daily_series(5)
        ref = pd.date_range("2022-01-01", periods=5, freq="D")
        result = align_to_reference(series, ref, method="ffill")
        np.testing.assert_array_equal(result.values, series.values)

    def test_mean_returns_series(self):
        series = _daily_series(30)
        ref = pd.date_range("2022-01-01", periods=30, freq="D")
        result = align_to_reference(series, ref, method="mean")
        assert isinstance(result, pd.Series)

    def test_mean_aligned_index_subset_of_ref(self):
        series = _daily_series(30)
        ref = pd.date_range("2022-01-01", periods=30, freq="D")
        result = align_to_reference(series, ref, method="mean")
        # All non-NaN entries must have index within ref
        valid = result.dropna()
        assert all(ts in ref for ts in valid.index)

    def test_unknown_method_raises_value_error(self):
        series = _daily_series(10)
        ref = pd.date_range("2022-01-01", periods=10, freq="D")
        with pytest.raises(ValueError, match="Unknown alignment method"):
            align_to_reference(series, ref, method="bad_method")


class TestEnsurePredictionAlignment:
    def test_same_length_returns_unchanged_values(self):
        preds = np.ones(10)
        y = pd.Series(np.zeros(10))
        result = ensure_prediction_alignment(preds, y)
        assert len(result) == 10
        np.testing.assert_array_equal(result.values, y.values)

    def test_shorter_preds_truncates_y_test(self):
        preds = np.ones(5)
        y = pd.Series(np.zeros(10))
        result = ensure_prediction_alignment(preds, y)
        assert len(result) == 5

    def test_longer_preds_truncates_to_y_test(self):
        preds = np.ones(15)
        y = pd.Series(np.zeros(10))
        result = ensure_prediction_alignment(preds, y)
        assert len(result) == 10

    def test_result_is_series(self):
        result = ensure_prediction_alignment(np.ones(5), pd.Series(range(5)))
        assert isinstance(result, pd.Series)

    def test_result_index_is_reset(self):
        y = pd.Series(range(5), index=[10, 20, 30, 40, 50])
        result = ensure_prediction_alignment(np.ones(5), y)
        assert list(result.index) == [0, 1, 2, 3, 4]

    def test_mismatch_does_not_raise(self):
        # Should handle mismatch gracefully without raising
        ensure_prediction_alignment(np.ones(7), pd.Series(np.zeros(3)))


class TestAlignTargetsAcrossIntervals:
    def _make_targets(self, n_daily=30, n_hourly=120):
        daily_idx = pd.date_range("2022-01-01", periods=n_daily, freq="D")
        hourly_idx = pd.date_range("2022-01-01", periods=n_hourly, freq="h")
        return {
            "1d": pd.Series(np.arange(float(n_daily)), index=daily_idx),
            "1h": pd.Series(np.arange(float(n_hourly)), index=hourly_idx),
        }

    def test_default_interval_is_unchanged(self):
        targets = self._make_targets()
        result = align_targets_across_intervals(targets, "1d", ["1d", "1h"])
        pd.testing.assert_series_equal(result["1d"], targets["1d"])

    def test_secondary_interval_aligned_to_ref_length(self):
        targets = self._make_targets(n_daily=30, n_hourly=120)
        result = align_targets_across_intervals(targets, "1d", ["1d", "1h"])
        assert len(result["1h"]) <= len(targets["1d"])

    def test_all_intervals_present_in_result(self):
        targets = self._make_targets()
        result = align_targets_across_intervals(targets, "1d", ["1d", "1h"])
        assert set(result.keys()) == {"1d", "1h"}

    def test_single_interval_passthrough(self):
        daily_idx = pd.date_range("2022-01-01", periods=20, freq="D")
        targets = {"1d": pd.Series(np.arange(20.0), index=daily_idx)}
        result = align_targets_across_intervals(targets, "1d", ["1d"])
        assert len(result) == 1
        pd.testing.assert_series_equal(result["1d"], targets["1d"])

    def test_result_indices_use_reference_index(self):
        targets = self._make_targets(n_daily=10, n_hourly=40)
        result = align_targets_across_intervals(targets, "1d", ["1d", "1h"])
        ref_idx = targets["1d"].index
        # The hourly result index must be a subset of the daily reference index
        for ts in result["1h"].index:
            assert ts in ref_idx
