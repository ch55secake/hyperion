from typing import override, Dict, Any

import pandas as pd

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

    @override
    @staticmethod
    def get_extra_params(prices_series: pd.Series) -> Dict[str, Any]:
        bb_period = 20
        bb_std = 2

        middle_band = prices_series.rolling(window=bb_period, min_periods=1).mean()
        rolling_std = prices_series.rolling(window=bb_period, min_periods=1).std()

        upper_band = middle_band + (rolling_std * bb_std)
        lower_band = middle_band - (rolling_std * bb_std)

        return {"upper_band": upper_band.to_dict(), "lower_band": lower_band.to_dict()}

    @override
    @staticmethod
    def get_minimum_data_points() -> int:
        return 20
