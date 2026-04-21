import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from src.util import logger


class RegimeDetector:
    """
    Classifies market into regimes by clustering rolling volatility and trend features.

    Typical usage:
        detector = RegimeDetector(n_regimes=2, vol_window=20)
        detector.fit(train_returns)
        regime = detector.predict_regime(recent_returns)
        all_labels = detector.label_regimes(returns_series)
    """

    def __init__(self, n_regimes: int = 2, vol_window: int = 20, random_state: int = 42):
        """
        :param n_regimes: number of market regimes to cluster into
        :param vol_window: rolling window length used to compute volatility / trend features
        :param random_state: random seed for KMeans reproducibility
        """
        self.n_regimes = n_regimes
        self.vol_window = vol_window
        self.random_state = random_state
        self._kmeans: KMeans | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_features(self, returns: pd.Series) -> pd.DataFrame:
        """Compute rolling-vol and rolling-mean features from a returns series."""
        df = pd.DataFrame(
            {
                "rolling_vol": returns.rolling(self.vol_window).std(),
                "rolling_mean": returns.rolling(self.vol_window).mean(),
            }
        )
        return df.dropna()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, returns: pd.Series) -> "RegimeDetector":
        """
        Fit the detector on historical returns.

        :param returns: pd.Series of periodic returns (e.g. daily close-to-close)
        :raises ValueError: if there are too few observations after removing NaNs
        """
        features = self._extract_features(returns)
        if len(features) < self.n_regimes:
            raise ValueError(
                f"Need at least {self.n_regimes} non-NaN feature rows to fit RegimeDetector, "
                f"got {len(features)}.  Try reducing vol_window or providing more data."
            )
        self._kmeans = KMeans(n_clusters=self.n_regimes, random_state=self.random_state, n_init=10)
        self._kmeans.fit(features.values)
        logger.info("RegimeDetector fitted with %d regimes on %d observations", self.n_regimes, len(features))
        return self

    def predict_regime(self, returns: pd.Series) -> int:
        """
        Classify the current market regime from recent returns.

        Uses the last available feature row (most recent rolling window) to assign a regime label.

        :param returns: pd.Series of recent returns (must contain at least vol_window values)
        :return: integer regime label in [0, n_regimes)
        :raises RuntimeError: if the detector has not been fitted
        """
        if self._kmeans is None:
            raise RuntimeError("RegimeDetector has not been fitted. Call fit() first.")
        features = self._extract_features(returns)
        if len(features) == 0:
            logger.warning("RegimeDetector: not enough data to extract features; defaulting to regime 0")
            return 0
        last_features = features.iloc[-1:].values
        return int(self._kmeans.predict(last_features)[0])

    def label_regimes(self, returns: pd.Series) -> pd.Series:
        """
        Assign a regime label to every timestamp in *returns* that has sufficient history.

        :param returns: pd.Series of returns indexed by timestamp
        :return: pd.Series[int] indexed by the same timestamps (NaN rows at the start are dropped)
        :raises RuntimeError: if the detector has not been fitted
        """
        if self._kmeans is None:
            raise RuntimeError("RegimeDetector has not been fitted. Call fit() first.")
        features = self._extract_features(returns)
        labels = pd.Series(
            self._kmeans.predict(features.values).astype(int),
            index=features.index,
            name="regime",
        )
        return labels
