"""Unit tests for src/align/align.py"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from src.align import align_targets_across_intervals, ensure_prediction_alignment
from src.align.align import align_to_reference

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _daily_index(start="2023-01-01", periods=10):
    return pd.date_range(start=start, periods=periods, freq="D")


def _series(index, values=None, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(0, 1, len(index)) if values is None else values
    return pd.Series(v, index=index)


def _daily_series(n, start="2022-01-01"):
    idx = pd.date_range(start, periods=n, freq="D")
    return pd.Series(np.arange(float(n)), index=idx)


# ---------------------------------------------------------------------------
# align_to_reference – ffill (temporal alignment regression tests)
# ---------------------------------------------------------------------------


class TestAlignToReferenceFfill:
    def test_temporal_alignment_exact_match(self):
        """When dates overlap exactly, values must be preserved."""
        idx = _daily_index(periods=5)
        series = _series(idx, values=[1.0, 2.0, 3.0, 4.0, 5.0])
        result = align_to_reference(series, idx, method="ffill")
        pd.testing.assert_series_equal(result, series)

    def test_temporal_alignment_subset_reference(self):
        """reference_index is a subset of targets dates – correct values selected."""
        full_idx = _daily_index(periods=10)
        values = list(range(10))
        series = _series(full_idx, values=values)
        ref = full_idx[:5]
        result = align_to_reference(series, ref, method="ffill")
        assert list(result.values) == values[:5]
        assert list(result.index) == list(ref)

    def test_temporal_alignment_shifted_dates(self):
        """targets_series starts 2 days after reference_index – first 2 entries should be NaN."""
        ref_idx = _daily_index(start="2023-01-01", periods=5)
        target_idx = _daily_index(start="2023-01-03", periods=5)
        series = _series(target_idx, values=[10.0, 20.0, 30.0, 40.0, 50.0])
        result = align_to_reference(series, ref_idx, method="ffill")
        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])
        assert result.iloc[2] == 10.0

    def test_no_positional_mispairing(self):
        """Positional slicing would pair wrong dates; temporal join must not do that."""
        ref_idx = _daily_index(start="2023-01-01", periods=5)
        target_idx = _daily_index(start="2023-01-03", periods=5)
        target_values = [100.0, 200.0, 300.0, 400.0, 500.0]
        series = _series(target_idx, values=target_values)
        result = align_to_reference(series, ref_idx, method="ffill")
        assert np.isnan(result["2023-01-01"])
        assert np.isnan(result["2023-01-02"])
        assert result["2023-01-03"] == 100.0

    def test_returns_series_with_reference_index(self):
        ref_idx = _daily_index(periods=5)
        series = _series(_daily_index(periods=8))
        result = align_to_reference(series, ref_idx, method="ffill")
        assert list(result.index) == list(ref_idx)

    def test_ffill_same_length_preserves_values(self):
        series = _daily_series(10)
        ref = pd.date_range("2022-01-01", periods=10, freq="D")
        result = align_to_reference(series, ref, method="ffill")
        assert len(result) == 10
        assert list(result.index) == list(ref)

    def test_ffill_series_shorter_than_ref_returns_ref_length(self):
        """With temporal alignment, result always matches reference_index length.
        Dates in ref beyond the series range are forward-filled from the last series value."""
        series = _daily_series(5)
        ref = pd.date_range("2022-01-01", periods=10, freq="D")
        result = align_to_reference(series, ref, method="ffill")
        assert len(result) == len(ref)

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

    def test_low_coverage_logs_warning(self):
        """Should log a warning when fewer than 50 % of reference dates are covered."""
        ref_idx = _daily_index(start="2023-01-01", periods=20)
        target_idx = _daily_index(start="2023-01-19", periods=2)
        series = _series(target_idx, values=[1.0, 2.0])
        with patch("src.align.align.logger") as mock_logger:
            align_to_reference(series, ref_idx, method="ffill")
            assert mock_logger.warning.called
            warning_msg = mock_logger.warning.call_args[0][0]
            assert "low coverage" in warning_msg.lower()


# ---------------------------------------------------------------------------
# align_to_reference – mean
# ---------------------------------------------------------------------------


class TestAlignToReferenceMean:
    def test_mean_returns_series_with_reference_index(self):
        ref_idx = _daily_index(start="2023-01-01", periods=5)
        hourly_idx = pd.date_range(start="2023-01-01", periods=5 * 24, freq="h")
        series = _series(hourly_idx)
        result = align_to_reference(series, ref_idx, method="mean")
        assert list(result.index) == list(ref_idx)

    def test_mean_returns_series(self):
        series = _daily_series(30)
        ref = pd.date_range("2022-01-01", periods=30, freq="D")
        result = align_to_reference(series, ref, method="mean")
        assert isinstance(result, pd.Series)

    def test_mean_aligned_index_subset_of_ref(self):
        series = _daily_series(30)
        ref = pd.date_range("2022-01-01", periods=30, freq="D")
        result = align_to_reference(series, ref, method="mean")
        valid = result.dropna()
        assert all(ts in ref for ts in valid.index)

    def test_fallback_daily_logs_warning_on_no_freq(self):
        """When freq cannot be inferred, a warning must be emitted."""
        irregular_idx = pd.DatetimeIndex(["2023-01-01", "2023-01-03", "2023-01-10"])
        target_idx = pd.date_range("2022-12-01", periods=60, freq="D")
        series = _series(target_idx)
        with patch("src.align.align.logger") as mock_logger:
            align_to_reference(series, irregular_idx, method="mean")
            assert mock_logger.warning.called
            warning_msg = mock_logger.warning.call_args[0][0]
            assert "falling back" in warning_msg.lower()

    def test_mean_low_coverage_logs_warning(self):
        ref_idx = _daily_index(start="2023-01-01", periods=20)
        target_idx = _daily_index(start="2023-06-01", periods=10)
        series = _series(target_idx, values=[float(i) for i in range(10)])
        with patch("src.align.align.logger") as mock_logger:
            align_to_reference(series, ref_idx, method="mean")
            assert mock_logger.warning.called
            warning_msg = mock_logger.warning.call_args[0][0]
            assert "low coverage" in warning_msg.lower()


# ---------------------------------------------------------------------------
# align_to_reference – unknown method
# ---------------------------------------------------------------------------


class TestAlignToReferenceUnknownMethod:
    def test_raises_on_unknown_method(self):
        idx = _daily_index(periods=5)
        series = _series(idx)
        with pytest.raises(ValueError, match="Unknown alignment method"):
            align_to_reference(series, idx, method="bad_method")

    def test_unknown_method_raises_value_error(self):
        series = _daily_series(10)
        ref = pd.date_range("2022-01-01", periods=10, freq="D")
        with pytest.raises(ValueError, match="Unknown alignment method"):
            align_to_reference(series, ref, method="bad_method")


# ---------------------------------------------------------------------------
# ensure_prediction_alignment
# ---------------------------------------------------------------------------


class TestEnsurePredictionAlignment:
    def test_no_truncation_when_lengths_match(self):
        idx = _daily_index(periods=10)
        y_test = _series(idx)
        preds = np.ones(10)
        result = ensure_prediction_alignment(preds, y_test)
        assert len(result) == 10

    def test_truncates_to_min_length(self):
        idx = _daily_index(periods=10)
        y_test = _series(idx)
        preds = np.ones(7)
        result = ensure_prediction_alignment(preds, y_test)
        assert len(result) == 7

    def test_logs_warning_on_mismatch(self):
        idx = _daily_index(periods=10)
        y_test = _series(idx)
        preds = np.ones(6)
        with patch("src.align.align.logger") as mock_logger:
            ensure_prediction_alignment(preds, y_test)
            assert mock_logger.warning.called
            warning_msg = mock_logger.warning.call_args[0][0]
            assert "mismatch" in warning_msg.lower()

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
        ensure_prediction_alignment(np.ones(7), pd.Series(np.zeros(3)))


# ---------------------------------------------------------------------------
# align_targets_across_intervals
# ---------------------------------------------------------------------------


class TestAlignTargetsAcrossIntervals:
    def _make_targets(self, n_daily=30, n_hourly=120):
        daily_idx = pd.date_range("2022-01-01", periods=n_daily, freq="D")
        hourly_idx = pd.date_range("2022-01-01", periods=n_hourly, freq="h")
        return {
            "1d": pd.Series(np.arange(float(n_daily)), index=daily_idx),
            "1h": pd.Series(np.arange(float(n_hourly)), index=hourly_idx),
        }

    def test_default_interval_unchanged(self):
        idx = _daily_index(periods=10)
        s = _series(idx)
        result = align_targets_across_intervals({"1d": s}, "1d", ["1d"])
        pd.testing.assert_series_equal(result["1d"], s)

    def test_default_interval_is_unchanged(self):
        targets = self._make_targets()
        result = align_targets_across_intervals(targets, "1d", ["1d", "1h"])
        pd.testing.assert_series_equal(result["1d"], targets["1d"])

    def test_non_default_interval_temporally_aligned(self):
        daily_idx = _daily_index(start="2023-01-01", periods=5)
        weekly_idx = _daily_index(start="2023-01-01", periods=5)
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        daily_series = _series(daily_idx, values=values)
        weekly_series = _series(weekly_idx, values=values)
        result = align_targets_across_intervals(
            {"1d": daily_series, "1w": weekly_series},
            "1d",
            ["1d", "1w"],
        )
        assert list(result["1w"].index) == list(daily_idx)
        assert list(result["1w"].values) == values

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
        for ts in result["1h"].index:
            assert ts in ref_idx
