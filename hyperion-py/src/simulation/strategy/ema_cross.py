from .strategy import Strategy
from .strategy_registry import register_strategy


@register_strategy("ema_cross")
class EMACrossStrategy(Strategy):
    def __init__(self, simulator, capital, ema_fast, ema_slow):
        super().__init__(simulator, capital)
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow

    def execute(self, date, price, pred_return, actual_return):
        f = self.ema_fast.get(date)
        s = self.ema_slow.get(date)

        if f is None or s is None:
            return self.capital, self.entry_price, self.position, self.shares

        if self.position is None and f > s:
            self.buy(date, price)

        elif self.position == "long" and f < s:
            self.sell(date, price, pred_return)

        return self.capital, self.entry_price, self.position, self.shares
