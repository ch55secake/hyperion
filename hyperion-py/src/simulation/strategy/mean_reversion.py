from .strategy import Strategy
from .strategy_registry import register_strategy


@register_strategy("bb_reversion")
class BollingerReversionStrategy(Strategy):
    def __init__(self, simulator, capital, lower_band, upper_band):
        super().__init__(simulator, capital)
        self.lower = lower_band
        self.upper = upper_band

    def execute(self, date, price, pred_return, actual_return):
        lb = self.lower.get(date)
        ub = self.upper.get(date)

        if lb is None or ub is None:
            return self.capital, self.entry_price, self.position, self.shares

        if self.position is None and price < lb:
            self.buy(date, price)

        elif self.position == "long" and price > ub:
            self.sell(date, price, pred_return)

        return self.capital, self.entry_price, self.position, self.shares
