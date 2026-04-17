from .strategy import Strategy
from .strategy_registry import register_strategy


@register_strategy("time_stop")
class TimeStopStrategy(Strategy):
    def __init__(self, simulator, capital, hold_days: int = 5):
        super().__init__(simulator, capital)
        self.hold_days = hold_days

    def execute(self, date, price, pred_return, actual_return):
        if self.position is None and pred_return > 0:
            self.buy(date, price)

        elif self.position == "long":
            self.hold_counter += 1
            if self.hold_counter >= self.hold_days:
                self.sell(date, price, pred_return)

        return self.capital, self.entry_price, self.position, self.shares
