from typing import Any

from .strategy import Strategy


class DirectionalTradingStrategy(Strategy):
    def __init__(self, simulator: Any, capital: Any, use_returns: bool = True):
        super().__init__(simulator, capital, use_returns)

    def execute(
        self, date: int | Any, price: int | Any, pred_return: Any, actual_return: Any
    ) -> tuple[int | float | Any, int | Any, Any, int | Any]:

        if self.position is None and pred_return > 0:
            self.buy(date, price, pred_return)

        elif self.position == "long" and pred_return <= 0:
            self.sell(date, price, pred_return, actual_return)

        return self.capital, self.entry_price, self.position, self.shares
