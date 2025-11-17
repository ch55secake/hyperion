from .strategy import Strategy
from .strategy_registry import register_strategy


@register_strategy("sltp")
class StopLossTakeProfitStrategy(Strategy):
    def __init__(self, simulator, capital, stop_loss=0.02, take_profit=0.04):
        super().__init__(simulator, capital)
        self.sl = stop_loss
        self.tp = take_profit

    def execute(self, date, price, pred_return, actual_return):
        if self.position is None and pred_return > 0:
            self.buy(date, price)

        elif self.position == "long":
            pnl = (price - self.entry_price) / self.entry_price

            if pnl <= -self.sl or pnl >= self.tp:
                self.sell(date, price, pred_return)

        return self.capital, self.entry_price, self.position, self.shares
