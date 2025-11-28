from typing import override, Dict, Any

import pandas as pd

from .strategy import Strategy
from .strategy_registry import register_strategy
from src.feature import std


@register_strategy("volatility_adjusted")
class VolatilityAdjustedStrategy(Strategy):
    def __init__(self, simulator, capital, k=0.5, vol_series=None):
        super().__init__(simulator, capital)
        self.k = k
        self.vol_series = vol_series

    def execute(self, date, price, pred_return, actual_return):
        vol = self.vol_series.get(date, 0.01)
        threshold = self.k * vol

        if self.position is None and pred_return > threshold:
            self.buy(date, price)

        elif self.position == "long" and pred_return < -threshold:
            self.sell(date, price, pred_return)

        return self.capital, self.entry_price, self.position, self.shares

    @override
    @staticmethod
    def get_extra_params(prices_series: pd.Series) -> Dict[str, Any]:
        vol_period = 20

        returns = prices_series.pct_change()
        vol_values = std(returns, vol_period)

        return {"vol_series": vol_values.to_dict()}

    @override
    @staticmethod
    def get_minimum_data_points() -> int:
        return 20
