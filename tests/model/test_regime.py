"""Unit tests for src/model/stacker/regime.py (RegimeDetector)."""

import numpy as np
import pandas as pd
import pytest

from src.model.stacker.regime import RegimeDetector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_returns(n=100, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="D")
    return pd.Series(rng.normal(0, 0.02, n), index=idx)


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestRegimeDetectorInit:
    def test_default_params(self):
        det = RegimeDetector()
        assert det.n_regimes == 2
        assert det.vol_window == 20
        assert det._kmeans is None

    def test_custom_params(self):
        det = RegimeDetector(n_regimes=3, vol_window=10)
        assert det.n_regimes == 3
        assert det.vol_window == 10


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------


class TestRegimeDetectorFit:
    def test_fit_sets_kmeans(self):
        det = RegimeDetector(n_regimes=2, vol_window=10)
        det.fit(_make_returns(n=80))
        assert det._kmeans is not None

    def test_fit_returns_self(self):
        det = RegimeDetector(n_regimes=2, vol_window=10)
        result = det.fit(_make_returns(n=80))
        assert result is det

    def test_fit_raises_with_too_few_rows(self):
        det = RegimeDetector(n_regimes=3, vol_window=20)
        # Only 5 rows of data — not enough to produce 3 feature rows after rolling
        with pytest.raises(ValueError, match="non-NaN feature rows"):
            det.fit(_make_returns(n=5))


# ---------------------------------------------------------------------------
# predict_regime
# ---------------------------------------------------------------------------


class TestPredictRegime:
    def _fitted_detector(self):
        det = RegimeDetector(n_regimes=2, vol_window=10)
        det.fit(_make_returns(n=80))
        return det

    def test_returns_int(self):
        det = self._fitted_detector()
        regime = det.predict_regime(_make_returns(n=50))
        assert isinstance(regime, int)

    def test_regime_within_valid_range(self):
        det = self._fitted_detector()
        regime = det.predict_regime(_make_returns(n=50))
        assert 0 <= regime < 2

    def test_raises_when_not_fitted(self):
        det = RegimeDetector()
        with pytest.raises(RuntimeError, match="fit()"):
            det.predict_regime(_make_returns(n=50))

    def test_returns_zero_for_insufficient_data(self):
        det = self._fitted_detector()
        # Series shorter than vol_window produces no feature rows
        regime = det.predict_regime(pd.Series([0.01, 0.02], index=pd.date_range("2022-01-01", periods=2, freq="D")))
        assert regime == 0


# ---------------------------------------------------------------------------
# label_regimes
# ---------------------------------------------------------------------------


class TestLabelRegimes:
    def _fitted_detector(self):
        det = RegimeDetector(n_regimes=2, vol_window=10)
        det.fit(_make_returns(n=80))
        return det

    def test_returns_series(self):
        det = self._fitted_detector()
        labels = det.label_regimes(_make_returns(n=80))
        assert isinstance(labels, pd.Series)

    def test_labels_are_valid_regime_ints(self):
        det = self._fitted_detector()
        labels = det.label_regimes(_make_returns(n=80))
        assert labels.dtype in (int, np.int32, np.int64) or np.issubdtype(labels.dtype, np.integer)
        assert labels.between(0, 1).all()

    def test_label_length_shorter_than_input(self):
        """Rows at the start lack enough history → are dropped by _extract_features."""
        det = self._fitted_detector()
        returns = _make_returns(n=80)
        labels = det.label_regimes(returns)
        assert len(labels) < len(returns)

    def test_series_name_is_regime(self):
        det = self._fitted_detector()
        labels = det.label_regimes(_make_returns(n=80))
        assert labels.name == "regime"

    def test_raises_when_not_fitted(self):
        det = RegimeDetector()
        with pytest.raises(RuntimeError, match="fit()"):
            det.label_regimes(_make_returns(n=50))
