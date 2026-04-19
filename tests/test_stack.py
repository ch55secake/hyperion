"""Unit tests for src/model/stacker/stack.py (TimeSeriesStacker)."""

import numpy as np
import pandas as pd
import pytest

from src.model.model import Model
from src.model.stacker.stack import TimeSeriesStacker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _LinearPredictor(Model):
    """Fits a simple mean predictor; fast enough for stacker unit tests."""

    def __init__(self, params=None):
        super().__init__("linear", params=params)
        self._mean = 0.0

    def train(self, x_train, y_train, x_val=None, y_val=None):
        self._prepare_columns(x_train)
        self.model = True
        valid = y_train.dropna() if hasattr(y_train, "dropna") else y_train[~np.isnan(y_train)]
        self._mean = float(np.mean(valid)) if len(valid) > 0 else 0.0

    def predict(self, x):
        x_proc = self._prepare_prediction(x)
        return np.full(len(x_proc), self._mean)


def _make_daily_data(n=100, n_features=3, seed=0):
    """Return (X DataFrame, y Series) with a daily DatetimeIndex."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="D")
    x = pd.DataFrame({f"f{i}": rng.normal(0, 1, n) for i in range(n_features)}, index=idx)
    y = pd.Series(rng.normal(0, 0.02, n), index=idx)
    return x, y


def _make_stacker(n=100, n_splits=3):
    x, y = _make_daily_data(n)
    meta_index = x.index

    base_models = [
        {
            "name": "model_a",
            "model_factory": _LinearPredictor,
            "X": x,
            "y": y,
            "align": "mean",
        },
        {
            "name": "model_b",
            "model_factory": _LinearPredictor,
            "X": x,
            "y": y,
            "align": "mean",
        },
    ]
    return TimeSeriesStacker(base_models=base_models, meta_index=meta_index, target=y, n_splits=n_splits)


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestTimeSeriesStackerInit:
    def test_meta_index_stored(self):
        stacker = _make_stacker()
        assert len(stacker.meta_index) == 100

    def test_target_aligned_to_meta_index(self):
        stacker = _make_stacker()
        assert len(stacker.target) == 100

    def test_default_meta_model_is_ridge(self):
        from sklearn.linear_model import Ridge

        stacker = _make_stacker()
        assert isinstance(stacker.meta_model, Ridge)

    def test_custom_meta_model_stored(self):
        from sklearn.linear_model import Lasso

        x, y = _make_daily_data(60)
        meta_index = x.index
        base = [{"name": "m", "model_factory": _LinearPredictor, "X": x, "y": y, "align": "mean"}]
        stacker = TimeSeriesStacker(base_models=base, meta_index=meta_index, target=y, meta_model=Lasso())
        assert isinstance(stacker.meta_model, Lasso)


# ---------------------------------------------------------------------------
# _make_time_splits
# ---------------------------------------------------------------------------


class TestMakeTimeSplits:
    def test_returns_correct_number_of_splits(self):
        stacker = _make_stacker(n_splits=3)
        splits = stacker._make_time_splits(50)
        assert len(splits) == 3

    def test_each_split_has_two_arrays(self):
        stacker = _make_stacker(n_splits=3)
        splits = stacker._make_time_splits(30)
        for train_idx, test_idx in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0


# ---------------------------------------------------------------------------
# _align_preds
# ---------------------------------------------------------------------------


class TestAlignPreds:
    def _make_preds(self, n=60, freq="D"):
        idx = pd.date_range("2022-01-01", periods=n, freq=freq)
        return pd.Series(np.arange(float(n)), index=idx)

    def test_mean_returns_series(self):
        stacker = _make_stacker()
        preds = self._make_preds()
        target_index = pd.date_range("2022-01-01", periods=60, freq="D")
        result = stacker._align_preds(preds, target_index, method="mean")
        assert isinstance(result, pd.Series)

    def test_ffill_returns_series(self):
        stacker = _make_stacker()
        preds = self._make_preds()
        target_index = pd.date_range("2022-01-01", periods=60, freq="D")
        result = stacker._align_preds(preds, target_index, method="ffill")
        assert isinstance(result, pd.Series)

    def test_ffill_result_length_matches_target(self):
        stacker = _make_stacker()
        preds = self._make_preds(30)
        target_index = pd.date_range("2022-01-01", periods=30, freq="D")
        result = stacker._align_preds(preds, target_index, method="ffill")
        assert len(result) == 30

    def test_callable_method_used(self):
        stacker = _make_stacker()
        preds = self._make_preds(20)
        target_index = pd.date_range("2022-01-01", periods=20, freq="D")
        called = {}

        def custom_align(p, idx):
            called["invoked"] = True
            return p.reindex(idx)

        stacker._align_preds(preds, target_index, method=custom_align)
        assert called.get("invoked")

    def test_unknown_string_method_raises(self):
        stacker = _make_stacker()
        preds = self._make_preds(20)
        target_index = pd.date_range("2022-01-01", periods=20, freq="D")
        with pytest.raises(ValueError, match="Unknown align method"):
            stacker._align_preds(preds, target_index, method="bogus")


# ---------------------------------------------------------------------------
# _oof_for_base – insufficient rows
# ---------------------------------------------------------------------------


class TestOofForBase:
    def test_insufficient_rows_raises_value_error(self):
        x_small, y_small = _make_daily_data(n=5)
        meta_index = x_small.index
        base = [{"name": "m", "model_factory": _LinearPredictor, "X": x_small, "y": y_small, "align": "mean"}]
        stacker = TimeSeriesStacker(base_models=base, meta_index=meta_index, target=y_small, n_splits=5)
        with pytest.raises(ValueError, match="Not enough rows"):
            stacker._oof_for_base(base[0])


# ---------------------------------------------------------------------------
# fit_meta
# ---------------------------------------------------------------------------


class TestFitMeta:
    def test_fit_meta_returns_dict_with_keys(self):
        stacker = _make_stacker(n=80, n_splits=3)
        result = stacker.fit_meta()
        assert "meta_oof_df" in result
        assert "meta_oof_preds" in result

    def test_oof_predictions_is_series(self):
        stacker = _make_stacker(n=80, n_splits=3)
        result = stacker.fit_meta()
        assert isinstance(result["meta_oof_preds"], pd.Series)

    def test_fitted_meta_set_after_fit_meta(self):
        stacker = _make_stacker(n=80, n_splits=3)
        stacker.fit_meta()
        assert stacker.fitted_meta is not None

    def test_meta_oof_df_has_base_model_columns(self):
        stacker = _make_stacker(n=80, n_splits=3)
        result = stacker.fit_meta()
        assert "model_a" in result["meta_oof_df"].columns
        assert "model_b" in result["meta_oof_df"].columns


# ---------------------------------------------------------------------------
# fit_full_and_predict
# ---------------------------------------------------------------------------


class TestFitFullAndPredict:
    def test_raises_if_fit_meta_not_called(self):
        stacker = _make_stacker(n=80, n_splits=3)
        test_index = pd.date_range("2022-01-01", periods=10, freq="D")
        with pytest.raises(RuntimeError, match="fit_meta"):
            stacker.fit_full_and_predict(test_index)

    def test_returns_base_preds_and_meta_preds(self):
        stacker = _make_stacker(n=80, n_splits=3)
        stacker.fit_meta()
        test_index = pd.date_range("2022-01-01", periods=80, freq="D")
        result = stacker.fit_full_and_predict(test_index)
        assert "base_preds" in result
        assert "meta_preds" in result

    def test_meta_preds_is_series(self):
        stacker = _make_stacker(n=80, n_splits=3)
        stacker.fit_meta()
        test_index = pd.date_range("2022-01-01", periods=80, freq="D")
        result = stacker.fit_full_and_predict(test_index)
        assert isinstance(result["meta_preds"], pd.Series)
