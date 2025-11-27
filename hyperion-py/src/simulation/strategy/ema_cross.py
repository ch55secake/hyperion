from typing import override, Dict, Any

import pandas as pd

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

    @override
    @staticmethod
    def get_extra_params(prices_series: pd.Series) -> Dict[str, Any]:
        ema_fast_period = 12
        ema_slow_period = 26

        ema_fast_values = prices_series.ewm(span=ema_fast_period, adjust=False).mean()
        ema_slow_values = prices_series.ewm(span=ema_slow_period, adjust=False).mean()

        return {
            "ema_fast": ema_fast_values.to_dict(),
            "ema_slow": ema_slow_values.to_dict(),
        }

    @override
    @staticmethod
    def get_minimum_data_points() -> int:
        return 26
