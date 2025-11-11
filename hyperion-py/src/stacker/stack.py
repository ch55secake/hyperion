import os
import pickle
from typing import Callable, Dict, List, Optional, Any
import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.base import clone


class TimeSeriesStacker:
    """
    TimeSeriesStacker: produce time-series-safe OOF predictions from multiple base models,
    train a meta-learner, and return final ensemble predictions.

    API:
      - base_models: list of dicts, each dict:
            {
              "name": "daily_xgb",
              "model_factory": lambda: XGBoostStockPredictor(...),
              "X": pd.DataFrame indexed by timestamps (base model features),
              "y": pd.Series indexed by same timestamps or can be None if not needed,
              "align": "mean" or "ffill" or callable(preds_series, target_index)->pd.Series
            }
          Note: base model's X,y should be for training the base model to predict the meta-target.
          If base model is trained on a different frequency (hourly), you can still provide X at hourly
          frequency and an align method to map its predictions to the meta index (e.g., daily).
      - meta_index: pd.DatetimeIndex for the meta-target (e.g., daily timestamps).
      - target: pd.Series indexed by meta_index (the actual target you want to predict; e.g., next-day return).
      - n_splits: number of TimeSeriesSplit folds to produce OOF predictions.
      - meta_model: scikit-learn regressor (default Ridge).
    """

    def __init__(
        self,
        base_models: List[Dict],
        meta_index: pd.DatetimeIndex,
        target: pd.Series,
        n_splits: int = 5,
        meta_model=None,
    ):
        self.base_models = base_models
        self.meta_index = meta_index
        self.target = target.loc[meta_index]
        self.n_splits = n_splits
        self.meta_model = meta_model if meta_model is not None else Ridge(alpha=1.0)
        self.oof_predictions = None
        self.meta_features = None
        self.trained_base_models = {}  # final models retrained on full data
        self.fitted_meta = None

    # ---------------------------
    # Helpers
    # ---------------------------
    def _align_preds(self, preds: pd.Series, target_index: pd.DatetimeIndex, method="mean"):
        """
        Align preds (indexed by timestamp at base model frequency) to the target_index.
        - method "mean": resample to freq of target_index and take mean within each target bin,
                         then reindex to target_index and forward-fill missing.
        - method "ffill": forward-fill to target_index using asof-style join.
        - method callable: call(preds, target_index) -> pd.Series
        """
        if callable(method):
            return method(preds, target_index)

        if method == "mean":
            # get the freq string for target_index (best-effort)
            # We'll aggregate preds into calendar days if target_index is daily
            # Use resample('D') if index appears daily, otherwise groupby target_index dates
            try:
                # try using target_index.freq if set
                freq = pd.infer_freq(target_index)
                if freq is not None:
                    # resample preds to that freq using mean
                    aligned = preds.resample(freq).mean()
                else:
                    # fallback: group by target date
                    aligned = preds.resample("D").mean()
            except Exception:
                aligned = preds.resample("D").mean()

            # Reindex to exact target_index and forward-fill last available within day
            aligned = aligned.reindex(target_index, method="ffill")
            return aligned

        elif method == "ffill":
            # forward-fill: align by asof - take last available prediction before target timestamp
            # pandas.merge_asof requires both as dataframes sorted
            s = preds.sort_index().rename("pred")
            t = pd.DataFrame(index=target_index)
            merged = pd.merge_asof(
                t.reset_index().rename(columns={"index": "target_time"}),
                s.reset_index().rename(columns={"index": "pred_time", "pred": "pred"}),
                left_on="target_time",
                right_on="pred_time",
                direction="backward",
            )
            res = pd.Series(merged["pred"].values, index=target_index)
            return res
        else:
            raise ValueError(f"Unknown align method: {method}")

    def _make_time_splits(self, idx_len):
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        return list(tscv.split(np.arange(idx_len)))

    # ---------------------------
    # OOF generation for one base model
    # ---------------------------
    def _oof_for_base(self, base: Dict):
        """
        base: dict with keys: name, model_factory, X (DataFrame), y (Series or None), align
        Returns: pandas.Series indexed by meta_index with OOF predictions aligned to meta_index
        """
        name = base["name"]
        model_factory: Callable = base["model_factory"]
        X: pd.DataFrame = base["X"].sort_index()
        y_base: Optional[pd.Series] = base.get("y", None)
        align = base.get("align", "mean")  # 'mean' or 'ffill' or callable

        # Build index for time splits based on X
        idx = X.index
        n = len(idx)
        if n < (self.n_splits + 1):
            raise ValueError(f"Not enough rows ({n}) in base model '{name}' to do {self.n_splits} splits")

        oof_preds_indexed = pd.Series(index=idx, dtype=float)

        # TimeSeriesSplit over X (so folds are based on base model frequency)
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        for train_idx, test_idx in tscv.split(np.arange(n)):
            train_idx_loc = idx[train_idx]
            test_idx_loc = idx[test_idx]

            X_tr = X.loc[train_idx_loc]
            y_tr = self.target.reindex(train_idx_loc) if y_base is None else y_base.loc[train_idx_loc]

            X_te = X.loc[test_idx_loc]
            # train a fresh model
            model = model_factory()
            # support models with train(X_train, y_train, x_val=None, y_val=None) signature
            try:
                model.train(X_tr, y_tr)
            except TypeError:
                # maybe model.train expects only X and y
                model.train(X_tr, y_tr)

            preds_te = pd.Series(model.predict(X_te), index=test_idx_loc)
            oof_preds_indexed.loc[test_idx_loc] = preds_te

        # Align the base-frequency OOF preds to meta_index
        aligned = self._align_preds(oof_preds_indexed.dropna(), self.meta_index, method=align)
        return aligned

    # ---------------------------
    # Fit stacker (produce OOF meta features, train meta)
    # ---------------------------
    def fit_meta(self):
        """
        Produce OOF predictions for all base models, build meta-features DataFrame (indexed by meta_index),
        and train the meta_model on rows where all base OOF features exist.
        """
        meta_df = pd.DataFrame(index=self.meta_index)

        print("Producing OOF predictions for base models...")
        for base in self.base_models:
            print(f"  -> OOF for {base['name']} ...")
            oof_aligned = self._oof_for_base(base)
            meta_df[base["name"]] = oof_aligned

        # drop rows with any missing base predictions
        meta_df = meta_df.dropna()
        y_meta = self.target.reindex(meta_df.index)

        print(f"Training meta model on {len(meta_df)} rows")
        self.meta_features = meta_df
        self.oof_predictions = pd.Series(self.meta_model.fit(meta_df, y_meta).predict(meta_df), index=meta_df.index)
        self.fitted_meta = clone(self.meta_model)
        self.fitted_meta.fit(meta_df, y_meta)  # store fitted meta learner

        print("Meta OOF R²:", r2_score(y_meta, self.oof_predictions))
        return {"meta_oof_df": meta_df, "meta_oof_preds": self.oof_predictions}

    # ---------------------------
    # Retrain base models on full data and predict on a test index
    # ---------------------------
    def fit_full_and_predict(self, test_meta_index: pd.DatetimeIndex):
        """
        Retrain each base model on full provided X (base['X']) and produce predictions aligned to test_meta_index.
        Then use fitted meta-learner to combine them.
        Returns dict with base_preds (DataFrame), meta_preds (Series), and evaluation if target available.
        """
        if self.fitted_meta is None:
            raise RuntimeError("Meta model not trained. Call fit_meta() first.")

        base_preds_df = pd.DataFrame(index=test_meta_index)
        for base in self.base_models:
            name = base["name"]
            print(f"Retraining base model '{name}' on full data...")
            model = base["model_factory"]()
            X_full = base["X"].sort_index()
            # base may or may not have y at its frequency; if missing, use meta target reindexed to base X index
            y_base = base.get("y", None)
            if y_base is None:
                y_full = self.target.reindex(X_full.index)
            else:
                y_full = y_base.reindex(X_full.index)

            model.train(X_full, y_full)
            self.trained_base_models[name] = model

            preds_full = pd.Series(model.predict(X_full), index=X_full.index)
            aligned = self._align_preds(preds_full, test_meta_index, method=base.get("align", "mean"))
            base_preds_df[name] = aligned

        base_preds_df = base_preds_df.dropna()
        meta_preds = pd.Series(self.fitted_meta.predict(base_preds_df), index=base_preds_df.index)

        # evaluation if test target available
        evals = {}
        if self.target is not None:
            y_test = self.target.reindex(meta_preds.index).dropna()
            common_idx = meta_preds.index.intersection(y_test.index)
            if len(common_idx) > 0:
                evals["r2"] = r2_score(y_test.loc[common_idx], meta_preds.loc[common_idx])
                evals["rmse"] = np.sqrt(mean_squared_error(y_test.loc[common_idx], meta_preds.loc[common_idx]))

        return {"base_preds": base_preds_df, "meta_preds": meta_preds, "evals": evals}


class StackedStockPredictor:
    """
    Stacked predictor combining multiple base models (e.g., daily + hourly).
    """

    def __init__(self, models: Dict[str, Any], weights: Dict[str, float] = None):
        """
        :param models: dict of models, e.g., {"daily": XGBoostStockPredictor(), "hourly": LightGBMStockPredictor()}
        :param weights: optional dict of weights for stacking
        """
        self.models = models
        self.weights = weights or {k: 1.0 for k in models}
        self.feature_importance = None

    def train(self, train_data: Dict[str, tuple]):
        """
        Train each base model.
        :param train_data: dict of (x_train, y_train, x_val, y_val) tuples per model
        """
        for name, model in self.models.items():
            x_train, y_train, x_val, y_val = train_data[name]
            model.train(x_train, y_train, x_val, y_val)
        self.feature_importance = self.compute_feature_importance()

    def predict(self, x_dict: dict) -> np.ndarray:
        """
        Return 1D stacked predictions
        """
        preds = []
        for name, model in self.models.items():
            p = model.predict(x_dict[name])
            # Ensure each model output is 1D
            p = np.asarray(p).ravel()
            preds.append(p * self.weights.get(name, 1.0))

        # Weighted average across models
        stacked_preds = np.sum(preds, axis=0) / sum(self.weights.values())

        # Make absolutely sure it's 1D
        return np.asarray(stacked_preds).ravel()

    def evaluate(self, x_dict: Dict[str, Any], y_true) -> dict:
        """
        Evaluate stacked model.
        """
        preds = self.predict(x_dict)

        mse = mean_squared_error(y_true, preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, preds)
        r2 = r2_score(y_true, preds)

        print(f"Stacked Model Performance:")
        print(f"  MSE : {mse:.8f}")
        print(f"  RMSE: {rmse:.8f}")
        print(f"  MAE : {mae:.8f}")
        print(f"  R²  : {r2:.8f}")

        return {
            "predictions": preds,  # ✅ 1D array
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
        }

    def save_model(self, symbol, save_path="models"):
        os.makedirs(save_path, exist_ok=True)
        filename = f"{save_path}/{symbol}_stacked_model.pkl"

        # Store feature column order for each base model
        feature_columns_per_model = {}
        for name, model in self.models.items():
            if hasattr(model, 'feature_columns') and model.feature_columns is not None:
                feature_columns_per_model[name] = model.feature_columns

        model_data = {
            'stacked_predictor': self,
            'feature_columns_per_model': feature_columns_per_model
        }

        with open(filename, "wb") as f:
            pickle.dump(model_data, f)
        print(f"✓ Saved full stacked model to {filename}")
        for name, cols in feature_columns_per_model.items():
            print(f"✓ Saved {len(cols)} feature columns for '{name}' model")

    @staticmethod
    def load_model(symbol, save_path="models"):
        filename = f"{save_path}/{symbol}_stacked_model.pkl"
        with open(filename, "rb") as f:
            model_data = pickle.load(f)

        # Handle both old format (just predictor) and new format (dict with metadata)
        if isinstance(model_data, dict) and 'stacked_predictor' in model_data:
            predictor = model_data['stacked_predictor']
            feature_columns_per_model = model_data.get('feature_columns_per_model', {})
            print(f"✓ Loaded stacked model from {filename}")
            for name, cols in feature_columns_per_model.items():
                print(f"✓ Model '{name}' expects {len(cols)} features in specific order")
        else:
            # Old format - just the predictor object
            predictor = model_data
            print(f"✓ Loaded stacked model from {filename} (old format)")

        return predictor

    def compute_feature_importance(self):
        combined = None
        for name, model in self.models.items():
            if hasattr(model, "feature_importance") and model.feature_importance is not None:
                fi = model.feature_importance.copy()
                weight = self.weights.get(name, 1.0)
                fi["importance"] *= weight
                fi.rename(columns={"importance": f"importance_{name}"}, inplace=True)
                if combined is None:
                    combined = fi
                else:
                    combined = combined.merge(fi, on="feature", how="outer")
        if combined is not None:
            # Fill NaNs with 0 and sum weighted importances
            combined = combined.fillna(0)
            importance_cols = [c for c in combined.columns if c.startswith("importance_")]
            combined["importance"] = combined[importance_cols].sum(axis=1)
            return combined[["feature", "importance"]].sort_values("importance", ascending=False)
        return None
