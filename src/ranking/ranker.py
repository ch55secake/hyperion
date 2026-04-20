from __future__ import annotations

import numpy as np
import pandas as pd

from src.util import logger


class StockRanker:
    """
    Ranks stocks by expected profitability using risk-adjusted metrics, confidence
    thresholds, and capital allocation rules.

    The pipeline follows these steps:
    1. Compute expected return from model predictions and current prices.
    2. Compute a normalized confidence score (0–1).
    3. Filter out low-confidence candidates.
    4. Estimate historical volatility as a risk proxy.
    5. Compute a priority score = (expected_return * confidence) / volatility.
    6. Rank stocks by priority score.
    7. Allocate capital proportionally to priority scores.
    """

    def __init__(
        self,
        min_confidence: float = 0.6,
        total_funds: float = 100_000.0,
        max_allocation_pct: float = 1.0,
        volatility_window: int = 20,
        min_volatility: float = 1e-6,
    ):
        """
        :param min_confidence: Minimum confidence score required to consider a trade (0–1).
        :param total_funds: Total capital available for allocation.
        :param max_allocation_pct: Maximum fraction of total_funds allocated to any single stock (0–1).
        :param volatility_window: Rolling window (in periods) used to compute historical volatility.
        :param min_volatility: Floor value applied to volatility to avoid division by zero.
        """
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")
        if total_funds <= 0:
            raise ValueError("total_funds must be positive")
        if not 0.0 < max_allocation_pct <= 1.0:
            raise ValueError("max_allocation_pct must be in (0, 1]")
        if volatility_window < 1:
            raise ValueError("volatility_window must be at least 1")
        if min_volatility <= 0:
            raise ValueError("min_volatility must be positive")

        self.min_confidence = min_confidence
        self.total_funds = total_funds
        self.max_allocation_pct = max_allocation_pct
        self.volatility_window = volatility_window
        self.min_volatility = min_volatility

    # ------------------------------------------------------------------
    # Step 1 – Expected return
    # ------------------------------------------------------------------

    @staticmethod
    def compute_expected_return(
        predicted_prices: pd.Series,
        current_prices: pd.Series,
    ) -> pd.Series:
        """Compute expected return from predicted and current prices.

        For regression models that predict returns directly, pass the predicted
        returns as *predicted_prices* and a Series of ones as *current_prices*
        (the formula simplifies to the predicted return itself).

        :param predicted_prices: Predicted future prices (or returns when current_prices=1).
        :param current_prices: Current market prices.
        :return: Series of expected returns.
        """
        current_prices = current_prices.replace(0, np.nan)
        return (predicted_prices - current_prices) / current_prices

    # ------------------------------------------------------------------
    # Step 2 – Confidence score
    # ------------------------------------------------------------------

    @staticmethod
    def compute_confidence_score(expected_returns: pd.Series) -> pd.Series:
        """Compute a normalized confidence score (0–1) for regression models.

        Larger absolute predicted returns imply higher confidence.  The score
        is normalised by the maximum absolute return in the batch so that the
        highest-confidence candidate always receives a score of 1.0.

        :param expected_returns: Series of expected returns (may be negative for shorts).
        :return: Series of confidence scores in [0, 1].
        """
        abs_returns = expected_returns.abs()
        max_abs = abs_returns.max()
        if max_abs == 0:
            return pd.Series(0.0, index=expected_returns.index)
        return abs_returns / max_abs

    # ------------------------------------------------------------------
    # Step 3 – Confidence threshold
    # ------------------------------------------------------------------

    def apply_confidence_threshold(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows whose confidence score is below *min_confidence*.

        :param df: DataFrame containing a ``confidence`` column.
        :return: Filtered DataFrame.  May be empty if no stock meets the threshold.
        """
        return df[df["confidence"] >= self.min_confidence].copy()

    # ------------------------------------------------------------------
    # Step 4 – Volatility
    # ------------------------------------------------------------------

    def compute_volatility(self, prices_by_symbol: dict[str, pd.Series]) -> dict[str, float]:
        """Compute historical volatility for each symbol.

        :param prices_by_symbol: Mapping of symbol → price Series (chronological order).
        :return: Mapping of symbol → volatility scalar.
        """
        volatilities: dict[str, float] = {}
        for symbol, prices in prices_by_symbol.items():
            returns = prices.pct_change().dropna()
            if len(returns) < 2:
                vol = self.min_volatility
            else:
                rolling_std = returns.rolling(self.volatility_window).std().dropna()
                if rolling_std.empty:
                    vol = max(returns.std(), self.min_volatility)
                else:
                    vol = max(rolling_std.iloc[-1], self.min_volatility)
                if np.isnan(vol):
                    vol = max(returns.std(), self.min_volatility)
            volatilities[symbol] = vol
        return volatilities

    # ------------------------------------------------------------------
    # Step 5 – Priority score
    # ------------------------------------------------------------------

    @staticmethod
    def compute_priority_score(
        expected_return: pd.Series,
        confidence: pd.Series,
        volatility: pd.Series,
    ) -> pd.Series:
        """Compute risk-adjusted priority score.

        priority = (expected_return * confidence) / volatility

        :param expected_return: Series of expected returns.
        :param confidence: Series of confidence scores in [0, 1].
        :param volatility: Series of volatility estimates (must be > 0).
        :return: Series of priority scores.
        """
        return (expected_return * confidence) / volatility

    # ------------------------------------------------------------------
    # Step 6 – Rank
    # ------------------------------------------------------------------

    @staticmethod
    def rank_stocks(df: pd.DataFrame) -> pd.DataFrame:
        """Sort DataFrame by ``priority_score`` in descending order.

        :param df: DataFrame with a ``priority_score`` column.
        :return: Sorted DataFrame with a ``rank`` column (1 = best).
        """
        ranked = df.sort_values(by="priority_score", ascending=False).copy()
        ranked["rank"] = range(1, len(ranked) + 1)
        return ranked

    # ------------------------------------------------------------------
    # Step 7 – Capital allocation
    # ------------------------------------------------------------------

    def allocate_capital(self, df: pd.DataFrame) -> pd.DataFrame:
        """Allocate *total_funds* proportionally to priority scores.

        Only stocks with a positive priority score receive capital.  An
        optional per-stock cap (``max_allocation_pct``) prevents excessive
        concentration.  When the cap is active, surplus capital is
        redistributed to uncapped stocks iteratively until stable.

        :param df: Ranked DataFrame with a ``priority_score`` column.
        :return: DataFrame with an ``allocation`` column added (dollars).
        """
        result = df.copy()
        positive_mask = result["priority_score"] > 0
        result["allocation"] = 0.0

        if not positive_mask.any():
            return result

        scores = result.loc[positive_mask, "priority_score"].copy()
        max_alloc = self.total_funds * self.max_allocation_pct
        allocations = pd.Series(0.0, index=scores.index)
        remaining_funds = self.total_funds
        remaining_scores = scores.copy()

        for _ in range(len(scores) + 1):
            if remaining_scores.empty or remaining_scores.sum() == 0:
                break

            total_score = remaining_scores.sum()
            raw = (remaining_scores / total_score) * remaining_funds

            overflow_mask = raw > max_alloc
            if not overflow_mask.any():
                allocations.loc[remaining_scores.index] = raw.values
                break

            capped_idx = remaining_scores.index[overflow_mask]
            allocations.loc[capped_idx] = max_alloc
            remaining_funds -= max_alloc * overflow_mask.sum()
            remaining_scores = remaining_scores[~overflow_mask]

        result.loc[positive_mask, "allocation"] = allocations.values
        return result

    # ------------------------------------------------------------------
    # Public orchestration method
    # ------------------------------------------------------------------

    def rank_and_allocate(
        self,
        predictions: dict[str, float],
        current_prices: dict[str, float],
        prices_history: dict[str, pd.Series] | None = None,
        historical_volatility: dict[str, float] | None = None,
        predictions_are_returns: bool = False,
    ) -> pd.DataFrame:
        """Run the full ranking and allocation pipeline.

        Either *prices_history* (raw OHLC-style price series per symbol) or a
        pre-computed *historical_volatility* mapping must be provided so that
        risk can be assessed.  If both are supplied, *historical_volatility*
        takes precedence.

        :param predictions: Mapping of symbol → predicted value.  When
                            ``predictions_are_returns=False`` (default) the values
                            are treated as predicted prices and expected return is
                            computed via ``(predicted - current) / current``.  When
                            ``predictions_are_returns=True`` the values are already
                            expected returns and are used directly.
        :param current_prices: Mapping of symbol → current market price.  Not used
                               when ``predictions_are_returns=True``.
        :param prices_history: Optional mapping of symbol → historical price Series
                               used to compute rolling volatility.
        :param historical_volatility: Optional pre-computed volatility mapping.
        :param predictions_are_returns: When ``True``, treat *predictions* as
                                        expected returns rather than predicted prices.
        :return: DataFrame with columns:
                 symbol, expected_return, confidence, volatility,
                 priority_score, rank, allocation.
                 Returns an empty DataFrame when no stock passes the
                 confidence threshold.
        """
        if not predictions:
            return pd.DataFrame(
                columns=[
                    "symbol",
                    "expected_return",
                    "confidence",
                    "volatility",
                    "priority_score",
                    "rank",
                    "allocation",
                ]
            )

        symbols = list(predictions.keys())
        pred_series = pd.Series(predictions, name="predicted")

        if predictions_are_returns:
            expected_returns = pred_series
        else:
            price_series = pd.Series({s: current_prices.get(s, 1.0) for s in symbols}, name="price")
            expected_returns = self.compute_expected_return(pred_series, price_series)

        confidence_scores = self.compute_confidence_score(expected_returns)

        df = pd.DataFrame(
            {
                "symbol": symbols,
                "expected_return": expected_returns.values,
                "confidence": confidence_scores.values,
            }
        )

        df = self.apply_confidence_threshold(df)

        if df.empty:
            logger.warning("⚠️  No stocks meet the confidence threshold – skipping trade execution.")
            return df.assign(volatility=np.nan, priority_score=np.nan, rank=np.nan, allocation=0.0)

        if historical_volatility is not None:
            df["volatility"] = df["symbol"].map(
                lambda s: max(historical_volatility.get(s, self.min_volatility), self.min_volatility)
            )
        elif prices_history is not None:
            vol_map = self.compute_volatility({s: prices_history[s] for s in df["symbol"] if s in prices_history})
            df["volatility"] = df["symbol"].map(lambda s: vol_map.get(s, self.min_volatility))
        else:
            df["volatility"] = self.min_volatility

        df["priority_score"] = self.compute_priority_score(
            pd.Series(df["expected_return"].values),
            pd.Series(df["confidence"].values),
            pd.Series(df["volatility"].values),
        ).values

        df = self.rank_stocks(df)
        df = self.allocate_capital(df)

        return df.reset_index(drop=True)
