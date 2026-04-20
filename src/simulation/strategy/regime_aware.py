"""Regime-aware trading strategy.

Routes execution to a momentum-style or mean-reversion-style sub-logic
depending on the current market regime supplied via a ``regime_series``
mapping (date → regime label).

Regime labels are expected to use the integer constants from
:mod:`src.feature.regime`:

* ``REGIME_BULL`` (0)     — trending bull market  → momentum logic
* ``REGIME_BEAR`` (1)     — trending bear market  → conservative, close longs
* ``REGIME_SIDEWAYS`` (2) — range-bound market    → mean-reversion logic

If the current date is not found in the regime mapping the strategy
falls back to a simple directional rule.
"""

from __future__ import annotations

from typing import Any, Dict, override

import pandas as pd

from src.feature.regime import (
    REGIME_BEAR,
    REGIME_BULL,
    REGIME_SIDEWAYS,
    classify_regime,
)
from src.simulation.strategy.strategy import Strategy
from src.simulation.strategy.strategy_registry import register_strategy


@register_strategy("regime_aware")
class RegimeAwareStrategy(Strategy):
    """Select momentum or mean-reversion logic based on detected market regime.

    Parameters
    ----------
    simulator:
        A :class:`~src.simulation.trading_simulator.TradingSimulator` instance.
    capital:
        Starting capital.
    regime_series:
        Mapping of ``date → regime_label`` (int).  Typically produced by
        :func:`~src.feature.regime.classify_regime` and converted to a dict
        via ``Series.to_dict()``.
    bull_entry_threshold:
        Minimum predicted return required to enter a long position in a bull
        regime.  Defaults to ``0.0`` (any positive prediction triggers a buy).
    reversion_entry_threshold:
        Minimum predicted return required to enter a long position in a
        sideways / mean-reversion regime.  A slightly higher bar avoids
        noise-driven trades in range-bound markets.
    """

    def __init__(
        self,
        simulator: Any,
        capital: Any,
        regime_series: Dict[Any, int],
        bull_entry_threshold: float = 0.0,
        reversion_entry_threshold: float = 0.005,
    ):
        super().__init__(simulator, capital)
        self.regime_series = regime_series
        self.bull_entry_threshold = bull_entry_threshold
        self.reversion_entry_threshold = reversion_entry_threshold

    # ------------------------------------------------------------------
    # Strategy.execute implementation
    # ------------------------------------------------------------------

    def execute(
        self,
        date: Any,
        price: Any,
        pred_return: Any,
        actual_return: Any,
    ) -> tuple[Any, Any, Any, Any]:
        regime = self.regime_series.get(date, REGIME_SIDEWAYS)

        if regime == REGIME_BULL:
            self._execute_momentum(date, price, pred_return)
        elif regime == REGIME_BEAR:
            self._execute_bear(date, price, pred_return)
        else:
            self._execute_reversion(date, price, pred_return)

        return self.capital, self.entry_price, self.position, self.shares

    # ------------------------------------------------------------------
    # Sub-logic helpers
    # ------------------------------------------------------------------

    def _execute_momentum(self, date: Any, price: Any, pred_return: Any) -> None:
        """Momentum-following logic for bull regimes."""
        if self.position is None and pred_return > self.bull_entry_threshold:
            self.buy(date, price)
        elif self.position == "long" and pred_return < 0:
            self.sell(date, price, pred_return)

    def _execute_bear(self, date: Any, price: Any, pred_return: Any) -> None:
        """Conservative logic for bear regimes: close longs early, no new entries."""
        if self.position == "long" and pred_return < 0:
            self.sell(date, price, pred_return)

    def _execute_reversion(self, date: Any, price: Any, pred_return: Any) -> None:
        """Mean-reversion logic for sideways regimes."""
        if self.position is None and pred_return > self.reversion_entry_threshold:
            self.buy(date, price)
        elif self.position == "long" and pred_return < -self.reversion_entry_threshold:
            self.sell(date, price, pred_return)

    # ------------------------------------------------------------------
    # Class-level helpers used by the strategy runner
    # ------------------------------------------------------------------

    @override
    @staticmethod
    def get_extra_params(prices_series: pd.Series) -> Dict[str, Any]:
        """Build a ``regime_series`` dict from raw prices."""
        returns = prices_series.pct_change(1).fillna(0)
        regime_df = classify_regime(prices_series, returns)
        return {"regime_series": regime_df["Regime_Trend"].to_dict()}

    @override
    @staticmethod
    def get_minimum_data_points() -> int:
        return 50

    # ------------------------------------------------------------------
    # Utility: regime label → human-readable name
    # ------------------------------------------------------------------

    @staticmethod
    def regime_name(label: int) -> str:
        return {REGIME_BULL: "bull", REGIME_BEAR: "bear", REGIME_SIDEWAYS: "sideways"}.get(label, "unknown")
