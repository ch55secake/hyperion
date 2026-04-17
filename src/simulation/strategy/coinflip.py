import random as rand

from .strategy import Strategy
from .strategy_registry import register_strategy


@register_strategy("coinflip")
class Coinflip(Strategy):
    """
    Josh hates the quant trading industry. 1% r2 feels stupid, and it upsets him greatly. He believes that a coinflip will
    always win. It doesn't.
    """

    def __init__(self, simulator, capital, threshold=-0.02):
        super().__init__(simulator, capital)
        self.threshold = threshold

    def execute(self, date, price, pred_return, actual_return):
        coin: int = rand.randint(0, 1)
        if self.position is None and coin == 1:
            self.buy(date, price)

        elif self.position == "long" and coin == 0:
            self.sell(date, price, pred_return)

        return self.capital, self.entry_price, self.position, self.shares
