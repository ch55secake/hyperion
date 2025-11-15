from typing import Any

from src.simulation.types import Trade, TradeAction

from .strategy import Strategy


class HoldDaysStrategy(Strategy):
    def __init__(self, simulator: Any, capital: Any, hold_days: int = 5, threshold: float = 0.02):
        super().__init__(simulator, capital)
        self.hold_days = hold_days
        self.threshold = threshold

    def execute(
        self, date: int | Any, price: int | Any, pred_return: Any, actual_return: Any
    ) -> tuple[int | float | Any, int | Any, Any, int | Any]:
        if self.position is None and pred_return > self.threshold:
            self.buy(date, price)

        elif self.position == "long":
            self.hold_counter += 1

            if self.hold_counter >= self.hold_days or pred_return < -self.threshold:
                self.sell(date, price, pred_return)

        return self.capital, self.entry_price, self.position, self.shares
