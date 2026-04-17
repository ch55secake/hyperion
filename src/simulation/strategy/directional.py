from typing import Any

from .strategy import Strategy
from .strategy_registry import register_strategy


@register_strategy("directional")
class DirectionalTradingStrategy(Strategy):
    def __init__(self, simulator: Any, capital: Any):
        super().__init__(simulator, capital)

    def execute(
        self, date: int | Any, price: int | Any, pred_return: Any, actual_return: Any
    ) -> tuple[int | float | Any, int | Any, Any, int | Any]:

        if self.position is None and pred_return > 0:
            self.buy(date, price)

        elif self.position == "long" and pred_return <= 0:
            self.sell(date, price, pred_return)

        return self.capital, self.entry_price, self.position, self.shares
