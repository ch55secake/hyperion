from __future__ import annotations

import re
from typing import Iterable, TypedDict

DEFAULT_SHARED_FEATURES: frozenset[str] = frozenset({"ticker", "sector", "industry"})


class FeaturePartition(TypedDict):
    daily: list[str]
    hourly: list[str]
    shared: list[str]
    per_column: dict[str, list[str]]


def _extract_windows(column: str) -> list[int]:
    return [int(match) for match in re.findall(r"(\d+)", column)]


def derive_feature_split(columns: Iterable[str], short_term_threshold: int = 20) -> FeaturePartition:
    """
    Split features into daily/hourly/shared buckets and document each column's assignment.
    """
    daily: set[str] = set()
    hourly: set[str] = set()
    shared: set[str] = set()
    partition_map: dict[str, set[str]] = {column: set() for column in columns}

    for column in columns:
        if column in DEFAULT_SHARED_FEATURES:
            shared.add(column)
            partition_map[column].add("shared")
            continue
        windows = _extract_windows(column)
        if not windows:
            shared.add(column)
            partition_map[column].add("shared")
            continue

        min_window = min(windows)
        max_window = max(windows)

        if max_window >= short_term_threshold:
            daily.add(column)
            partition_map[column].add("daily")
        if min_window < short_term_threshold:
            hourly.add(column)
            partition_map[column].add("hourly")
        if column not in daily and column not in hourly:
            shared.add(column)
            partition_map[column].add("shared")

    for column, buckets in partition_map.items():
        if not buckets:
            shared.add(column)
            buckets.add("shared")

    partition: FeaturePartition = {
        "daily": sorted(daily),
        "hourly": sorted(hourly),
        "shared": sorted(shared),
        "per_column": {column: sorted(buckets) for column, buckets in partition_map.items()},
    }
    return partition
