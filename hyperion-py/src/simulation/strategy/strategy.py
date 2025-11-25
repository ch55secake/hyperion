from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any

from src.simulation.types import TradeAction, Trade


class Strategy(ABC):
    def __init__(self, simulator: Any, capital: float | Any):
        self.simulator = simulator
        self.capital = capital
        self.position = None
        self.entry_price: int | Any = 0
        self.shares: int | float = 0
        self.hold_counter: int = 0

    @abstractmethod
    def execute(
        self, date: int | Any, price: int | Any, pred_return: Any, actual_return: Any
    ) -> tuple[int | float | Any, int | Any, Any, int | Any]:
        pass

    def buy(self, date: int | Any, price: int | Any) -> None:
        self.shares = (self.capital * (1 - self.simulator.transaction_cost)) / price
        self.entry_price = price
        self.capital = 0

        self.position = "long"

        self.simulator.trades.append(
            Trade(
                date=date,
                action=TradeAction.BUY.value,
                predicted_return=None,
                price=price,
                pnl_pct=None,
                profit=None,
            )
        )

    def sell(self, date: int | Any, price: int | Any, pred_return: Any) -> None:
        self.capital = self.shares * price * (1 - self.simulator.transaction_cost)
        profit = self.capital - self.simulator.initial_capital
        pnl = ((price - self.entry_price) / self.entry_price) * 100

        self.simulator.trades.append(
            Trade(
                date=date,
                action=TradeAction.SELL.value,
                predicted_return=pred_return,
                price=price,
                profit=profit,
                pnl_pct=pnl,
            )
        )

        self.position = None
        self.shares = 0
        self.hold_counter = 0

    @staticmethod
    def simulate(strategy: Strategy, dates_test, prices_test, predictions, y_test):
        # Use the simulator from the passed strategy instance
        sim_results = strategy.simulator.simulate(
            predictions, y_test, prices_test, dates_test, threshold=0, strategy=strategy
        )

        return sim_results, strategy.simulator
