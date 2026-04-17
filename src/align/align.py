import pandas as pd


def align_to_reference(targets_series, reference_index, method="ffill"):
    """
    Align targets from one interval to a reference index.
    :param targets_series: Target series to align
    :param reference_index: Reference index to align to
    :param method: Alignment method ("ffill", "mean")
    :return: Aligned targets series
    """
    if method == "ffill":
        min_length = min(len(targets_series), len(reference_index))

        aligned_index = reference_index[:min_length]
        aligned_values = targets_series.iloc[:min_length].values

        return pd.Series(aligned_values, index=aligned_index)

    elif method == "mean":
        try:
            freq = pd.infer_freq(reference_index)
            if freq is not None:
                resampled = targets_series.resample(freq).mean()
            else:
                resampled = targets_series.resample("D").mean()
        except Exception:
            resampled = targets_series.resample("D").mean()

        aligned = resampled.reindex(reference_index, method="ffill")
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
        print(
            f"Warning: Prediction length ({len(predictions)}) and y_test length ({len(y_test)}) mismatch. Using {min_len} samples."
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
