"""Unit tests for src/align/align.py"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from src.align.align import align_targets_across_intervals, align_to_reference, ensure_prediction_alignment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _daily_index(start="2023-01-01", periods=10):
    return pd.date_range(start=start, periods=periods, freq="D")


def _series(index, values=None, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(0, 1, len(index)) if values is None else values
    return pd.Series(v, index=index)


# ---------------------------------------------------------------------------
# align_to_reference – ffill
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
        # Reference covers only the first 5 days
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
        # 2023-01-01 and 2023-01-02 have no preceding target value → NaN
        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])
        # 2023-01-03 onwards should match
        assert result.iloc[2] == 10.0

    def test_no_positional_mispairing(self):
        """Positional slicing would pair wrong dates; temporal join must not do that."""
        # target: Jan 3-7, reference: Jan 1-5
        ref_idx = _daily_index(start="2023-01-01", periods=5)
        target_idx = _daily_index(start="2023-01-03", periods=5)
        target_values = [100.0, 200.0, 300.0, 400.0, 500.0]
        series = _series(target_idx, values=target_values)
        result = align_to_reference(series, ref_idx, method="ffill")
        # Positional alignment would place 100 at Jan-01; temporal won't
        assert np.isnan(result["2023-01-01"])
        assert np.isnan(result["2023-01-02"])
        assert result["2023-01-03"] == 100.0

    def test_returns_series_with_reference_index(self):
        ref_idx = _daily_index(periods=5)
        series = _series(_daily_index(periods=8))
        result = align_to_reference(series, ref_idx, method="ffill")
        assert list(result.index) == list(ref_idx)

    def test_low_coverage_logs_warning(self):
        """Should log a warning when fewer than 50 % of reference dates are covered."""
        ref_idx = _daily_index(start="2023-01-01", periods=20)
        # targets only exist on the last 2 days of the reference range
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

    def test_fallback_daily_logs_warning_on_no_freq(self):
        """When freq cannot be inferred, a warning must be emitted."""
        # Irregular index → infer_freq returns None
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
        # targets only in June – all dates are AFTER reference range, so ffill can't fill back
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


# ---------------------------------------------------------------------------
# align_targets_across_intervals
# ---------------------------------------------------------------------------


class TestAlignTargetsAcrossIntervals:
    def test_default_interval_unchanged(self):
        idx = _daily_index(periods=10)
        s = _series(idx)
        result = align_targets_across_intervals({"1d": s}, "1d", ["1d"])
        pd.testing.assert_series_equal(result["1d"], s)

    def test_non_default_interval_temporally_aligned(self):
        daily_idx = _daily_index(start="2023-01-01", periods=5)
        # Same dates – temporal join should return identical values
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

