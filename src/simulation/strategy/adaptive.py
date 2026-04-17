from typing import Any

from .strategy import Strategy
from .strategy_registry import register_strategy


@register_strategy("adaptive")
class AdaptiveThresholdStrategy(Strategy):
    def __init__(self, simulator: Any, capital: Any, threshold: float = 0.02):
        super().__init__(simulator, capital)
        self.threshold = threshold

    def execute(
        self, date: int | Any, price: int | Any, pred_return: Any, actual_return: Any
    ) -> tuple[int | float | Any, int | Any, Any, int | Any]:

        if self.position is None and pred_return > self.threshold:
            self.buy(date, price)

        elif pred_return < -self.threshold:
            self.sell(date, price, pred_return)

        return self.capital, self.entry_price, self.position, self.shares
