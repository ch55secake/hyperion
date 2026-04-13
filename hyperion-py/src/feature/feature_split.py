from __future__ import annotations

import re
from typing import Iterable, TypedDict

DEFAULT_SHARED_FEATURES: frozenset[str] = frozenset({"ticker", "sector", "industry"})

# Categorical/metadata features always included in every model
_ALWAYS_SHARED: frozenset[str] = frozenset({"ticker", "sector", "industry", "beta", "avg_volume_log", "market_cap"})

# Prefix/name patterns that identify technical-indicator features (Model 1 – XGBoost)
_TECHNICAL_PREFIXES: tuple[str, ...] = (
    "SMA_",
    "EMA_",
    "WMA_",
    "HMA_",
    "Price_to_SMA",
    "Price_to_EMA",
    "EMA_12_26_Ratio",
    "SMA_5_20_Ratio",
    "SMA_10_50_Ratio",
    "MACD",
    "RSI",
    "BB_",
    "CCI",
    "WilliamsR",
    "Stochastic",
    "TSI",
    "Plus_DI",
    "Minus_DI",
    "ADX",
    "Bull_Engulfing",
    "Doji",
)

# Prefix/name patterns that identify price-action features (Model 2 – LightGBM)
_PRICE_ACTION_PREFIXES: tuple[str, ...] = (
    "Price_Change_",
    "Return_",
    "Volatility_",
    "Volume_",
    "HL_Range",
    "Sharpe_",
)

# Prefix/name patterns that identify momentum/microstructure features (Model 3 – CatBoost)
_MOMENTUM_PREFIXES: tuple[str, ...] = (
    "Momentum_",
    "ROC_",
    "ATR",
)


class FeaturePartition(TypedDict):
    daily: list[str]
    hourly: list[str]
    shared: list[str]
    per_column: dict[str, list[str]]


class ModelFeatureSplit(TypedDict):
    technical_indicators: list[str]
    price_action: list[str]
    momentum: list[str]
    shared: list[str]
    per_column: dict[str, str]


def _extract_windows(column: str) -> list[int]:
    return [int(match) for match in re.findall(r"(\d+)", column)]


def _classify_feature(column: str) -> str:
    """Return the model group ('technical_indicators', 'price_action', 'momentum', or 'shared') for a column."""
    if column in _ALWAYS_SHARED:
        return "shared"
    for prefix in _TECHNICAL_PREFIXES:
        if column.startswith(prefix) or column == prefix.rstrip("_"):
            return "technical_indicators"
    for prefix in _PRICE_ACTION_PREFIXES:
        if column.startswith(prefix) or column == prefix.rstrip("_"):
            return "price_action"
    for prefix in _MOMENTUM_PREFIXES:
        if column.startswith(prefix) or column == prefix.rstrip("_"):
            return "momentum"
    return "shared"


def derive_model_feature_split(columns: Iterable[str]) -> ModelFeatureSplit:
    """
    Split features into three semantically distinct groups to maximise model diversity:

    - ``technical_indicators``: oscillators and trend-following indicators (RSI, MACD, Bollinger
      Bands, moving averages, …).  Used by Model 1 (XGBoost).
    - ``price_action``: returns, volatility, volume and high-low range metrics.
      Used by Model 2 (LightGBM).
    - ``momentum``: rate-of-change, momentum and ATR features.
      Used by Model 3 (CatBoost).
    - ``shared``: categorical / metadata columns included in every model.

    Each base model receives its own group *plus* the shared features so there is
    a small, intentional overlap (metadata) while keeping the predictive features
    distinct.
    """
    technical: list[str] = []
    price_action: list[str] = []
    momentum: list[str] = []
    shared: list[str] = []
    per_column: dict[str, str] = {}

    for column in columns:
        group = _classify_feature(column)
        per_column[column] = group
        if group == "technical_indicators":
            technical.append(column)
        elif group == "price_action":
            price_action.append(column)
        elif group == "momentum":
            momentum.append(column)
        else:
            shared.append(column)

    return ModelFeatureSplit(
        technical_indicators=sorted(technical),
        price_action=sorted(price_action),
        momentum=sorted(momentum),
        shared=sorted(shared),
        per_column=per_column,
    )


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
