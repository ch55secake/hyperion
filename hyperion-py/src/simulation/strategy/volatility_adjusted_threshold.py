from .strategy import Strategy
from .strategy_registry import register_strategy


@register_strategy("volatility_adjusted")
class VolatilityAdjustedStrategy(Strategy):
    def __init__(self, simulator, capital, k=0.5, vol_series=None):
        super().__init__(simulator, capital)
        self.k = k
        self.vol_series = vol_series  # pass rolling std for each timestep

    def execute(self, date, price, pred_return, actual_return):
        vol = self.vol_series.get(date, 0.01)
        threshold = self.k * vol

        if self.position is None and pred_return > threshold:
            self.buy(date, price)

        elif self.position == "long" and pred_return < -threshold:
            self.sell(date, price, pred_return)

        return self.capital, self.entry_price, self.position, self.shares
