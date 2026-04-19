import pandas as pd

from src.util import logger


def align_to_reference(targets_series, reference_index, method="ffill"):
    """
    Align targets from one interval to a reference index using temporal (date-based) alignment.
    :param targets_series: Target series to align
    :param reference_index: Reference index to align to
    :param method: Alignment method ("ffill", "mean")
    :return: Aligned targets series
    """
    if method == "ffill":
        aligned = targets_series.reindex(reference_index, method="ffill")

        non_null = aligned.notna().sum()
        coverage = non_null / len(reference_index) if len(reference_index) > 0 else 0.0
        if coverage < 0.5:
            logger.warning(
                f"align_to_reference (ffill): low coverage {coverage * 100:.1f}% "
                f"({non_null}/{len(reference_index)} dates matched). "
                "Check that targets_series and reference_index share overlapping dates."
            )

        return aligned

    elif method == "mean":
        try:
            freq = pd.infer_freq(reference_index)
            if freq is not None:
                resampled = targets_series.resample(freq).mean()
            else:
                logger.warning(
                    "align_to_reference (mean): could not infer frequency from reference_index. "
                    "Falling back to daily ('D') resampling."
                )
                resampled = targets_series.resample("D").mean()
        except Exception as exc:
            logger.warning(
                f"align_to_reference (mean): error during frequency detection or resampling ({exc}). "
                "Falling back to daily ('D') resampling."
            )
            resampled = targets_series.resample("D").mean()

        aligned = resampled.reindex(reference_index, method="ffill")

        non_null = aligned.notna().sum()
        coverage = non_null / len(reference_index) if len(reference_index) > 0 else 0.0
        if coverage < 0.5:
            logger.warning(
                f"align_to_reference (mean): low coverage {coverage * 100:.1f}% "
                f"({non_null}/{len(reference_index)} dates matched). "
                "Check that targets_series and reference_index share overlapping dates."
            )

        return aligned

    else:
        raise ValueError(f"Unknown alignment method: {method}")


def ensure_prediction_alignment(predictions, y_test):
    """
    Ensure y_test aligns with prediction array indices.
    :param predictions: predictions array that we want to align
    :return: aligned y_test series
    """
    if len(predictions) != len(y_test):
        min_len = min(len(predictions), len(y_test))
        logger.warning(
            "Prediction length (%d) and y_test length (%d) mismatch. Using %d samples.",
            len(predictions),
            len(y_test),
            min_len,
        )
        aligned_y_test = y_test.iloc[:min_len].reset_index(drop=True)
    else:
        aligned_y_test = y_test.reset_index(drop=True)

    return aligned_y_test


def align_targets_across_intervals(y_test_dict, default_interval, intervals):
    """
    Align y_test targets across different time intervals to ensure consistency.
    Uses the default interval as the reference timeline.
    :return: the aligned targets per interval
    """
    aligned_targets = {}

    reference_index = y_test_dict[default_interval].index

    for interval in intervals:
        if interval == default_interval:
            aligned_targets[interval] = y_test_dict[interval]
        else:
            aligned_targets[interval] = align_to_reference(y_test_dict[interval], reference_index, method="ffill")

    return aligned_targets
