from .strategy import Strategy
from .strategy_registry import register_strategy


@register_strategy("sma_trend")
class SMATrendStrategy(Strategy):
    def __init__(self, simulator, capital, sma_series):
        super().__init__(simulator, capital)
        self.sma_series = sma_series

    def execute(self, date, price, pred_return, actual_return):
        sma = self.sma_series.get(date)

        if sma is None:
            return self.capital, self.entry_price, self.position, self.shares

        if self.position is None and pred_return > 0 and price > sma:
            self.buy(date, price)

        elif self.position == "long" and price < sma:
            self.sell(date, price, pred_return)

        return self.capital, self.entry_price, self.position, self.shares
