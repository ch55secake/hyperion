"""
Market regime detection utilities.

Three complementary regime signals are provided:

* **Volatility regime** — classifies rolling realised volatility into
  low / medium / high buckets using rolling-quantile thresholds.

* **Trend regime** — compares the closing price to a fast and a slow
  moving average to identify bull / bear / sideways markets.

* **HMM regime** — fits a Gaussian Mixture Model (a tractable
  approximation of a Hidden Markov Model) on rolling return features and
  labels each period with the most likely latent state, sorted so that
  state 0 is always the most bearish.

All three signals are assembled by :func:`classify_regime`, which returns a
``DataFrame`` that can be merged directly into a feature matrix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

# ---------------------------------------------------------------------------
# Integer labels
# ---------------------------------------------------------------------------

REGIME_BULL = 0
REGIME_BEAR = 1
REGIME_SIDEWAYS = 2

REGIME_VOL_LOW = 0
REGIME_VOL_MEDIUM = 1
REGIME_VOL_HIGH = 2

# Column names emitted by classify_regime
COL_REGIME_VOLATILITY = "Regime_Volatility"
COL_REGIME_TREND = "Regime_Trend"
COL_REGIME_HMM = "Regime_HMM"

_MIN_SAMPLES_FOR_GMM = 10


def detect_volatility_regime(
    returns: pd.Series,
    window: int = 20,
    low_vol_percentile: float = 33.0,
    high_vol_percentile: float = 67.0,
) -> pd.Series:
    """Classify rolling volatility into low / medium / high regimes.

    Parameters
    ----------
    returns:
        Daily (or intraday) return series.
    window:
        Rolling window used to compute realised volatility.
    low_vol_percentile:
        Percentile below which volatility is classified as *low*.
    high_vol_percentile:
        Percentile above which volatility is classified as *high*.

    Returns
    -------
    pd.Series of dtype int8 with values:
        ``REGIME_VOL_LOW`` (0), ``REGIME_VOL_MEDIUM`` (1), or
        ``REGIME_VOL_HIGH`` (2).
    """
    rolling_vol = returns.rolling(window, min_periods=1).std().fillna(0)
    low_threshold = float(rolling_vol.quantile(low_vol_percentile / 100))
    high_threshold = float(rolling_vol.quantile(high_vol_percentile / 100))

    regime = pd.Series(REGIME_VOL_MEDIUM, index=returns.index, dtype=np.int8)
    regime[rolling_vol <= low_threshold] = REGIME_VOL_LOW
    regime[rolling_vol >= high_threshold] = REGIME_VOL_HIGH
    return regime


def detect_trend_regime(
    close: pd.Series,
    fast_ma: int = 50,
    slow_ma: int = 200,
) -> pd.Series:
    """Classify trend regime using price relative to moving averages.

    * **Bull** — price > fast MA *and* fast MA > slow MA.
    * **Bear** — price < fast MA *and* fast MA < slow MA.
    * **Sideways** — any other configuration.

    Parameters
    ----------
    close:
        Closing price series.
    fast_ma:
        Window for the fast (shorter-term) moving average.
    slow_ma:
        Window for the slow (longer-term) moving average.

    Returns
    -------
    pd.Series of dtype int8 with values ``REGIME_BULL`` (0),
    ``REGIME_BEAR`` (1), or ``REGIME_SIDEWAYS`` (2).
    """
    n = len(close)
    fast = close.rolling(min(fast_ma, n), min_periods=1).mean()
    slow = close.rolling(min(slow_ma, n), min_periods=1).mean()

    regime = pd.Series(REGIME_SIDEWAYS, index=close.index, dtype=np.int8)
    regime[(close > fast) & (fast > slow)] = REGIME_BULL
    regime[(close < fast) & (fast < slow)] = REGIME_BEAR
    return regime


def detect_hmm_regime(
    returns: pd.Series,
    n_states: int = 2,
    random_state: int = 42,
) -> pd.Series:
    """Detect latent regimes via a Gaussian Mixture Model on rolling features.

    The model is fitted on two rolling statistics (mean return and realised
    volatility) computed over a 20-period window.  States are relabelled so
    that state 0 always has the *lowest* mean return (most bearish) and
    state ``n_states - 1`` has the highest (most bullish), making the labels
    consistent across different datasets.

    If too few samples are available the function falls back to a single
    "unknown" regime (all zeros).

    Parameters
    ----------
    returns:
        Return series (e.g. ``Close.pct_change()``).
    n_states:
        Number of latent states.  Must be ≥ 2.
    random_state:
        Random seed passed to :class:`~sklearn.mixture.GaussianMixture`.

    Returns
    -------
    pd.Series of dtype int8 with values in ``[0, n_states - 1]``.
    """
    n_states = max(2, n_states)
    window = 20
    rolling_mean = returns.rolling(window, min_periods=1).mean().fillna(0)
    rolling_std = returns.rolling(window, min_periods=1).std().fillna(0)

    features = np.column_stack([rolling_mean.values, rolling_std.values])

    if len(features) < _MIN_SAMPLES_FOR_GMM:
        return pd.Series(np.zeros(len(returns), dtype=np.int8), index=returns.index)

    gmm = GaussianMixture(
        n_components=n_states,
        covariance_type="full",
        random_state=random_state,
        n_init=3,
    )
    raw_labels = gmm.fit_predict(features)

    # Sort by mean return so that label 0 == most bearish
    component_means = gmm.means_[:, 0]
    order = np.argsort(component_means)
    remap = {int(old): int(new) for new, old in enumerate(order)}
    labels = np.array([remap[int(lbl)] for lbl in raw_labels], dtype=np.int8)

    return pd.Series(labels, index=returns.index, dtype=np.int8)


def classify_regime(
    close: pd.Series,
    returns: pd.Series,
    fast_ma: int = 50,
    slow_ma: int = 200,
    vol_window: int = 20,
    n_hmm_states: int = 2,
) -> pd.DataFrame:
    """Compute all regime features and return them as a DataFrame.

    This is the primary entry point for regime detection.  It computes
    all three regime signals and aligns them on the index of ``close``.

    Parameters
    ----------
    close:
        Closing price series (must share its index with ``returns``).
    returns:
        Return series (e.g. ``close.pct_change()``).
    fast_ma:
        Fast MA window for trend regime detection.
    slow_ma:
        Slow MA window for trend regime detection.
    vol_window:
        Rolling window for volatility regime detection.
    n_hmm_states:
        Number of latent states for the GMM-based regime.

    Returns
    -------
    pd.DataFrame with columns:
        * ``Regime_Volatility`` — 0 (low) / 1 (medium) / 2 (high)
        * ``Regime_Trend``      — 0 (bull) / 1 (bear) / 2 (sideways)
        * ``Regime_HMM``        — 0 (most bearish) … n_hmm_states-1 (most bullish)
    """
    vol_regime = detect_volatility_regime(returns, window=vol_window)
    trend_regime = detect_trend_regime(close, fast_ma=fast_ma, slow_ma=slow_ma)
    hmm_regime = detect_hmm_regime(returns, n_states=n_hmm_states)

    return pd.DataFrame(
        {
            COL_REGIME_VOLATILITY: vol_regime,
            COL_REGIME_TREND: trend_regime,
            COL_REGIME_HMM: hmm_regime,
        },
        index=close.index,
    )
