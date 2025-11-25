from .strategy import Strategy
from .strategy_registry import register_strategy


@register_strategy("contrarian")
class ContrarianStrategy(Strategy):
    def __init__(self, simulator, capital, threshold=-0.02):
        super().__init__(simulator, capital)
        self.threshold = threshold

    def execute(self, date, price, pred_return, actual_return):
        if self.position is None and pred_return < self.threshold:
            self.buy(date, price)

        elif self.position == "long" and pred_return > -self.threshold:
            self.sell(date, price, pred_return)

        return self.capital, self.entry_price, self.position, self.shares
