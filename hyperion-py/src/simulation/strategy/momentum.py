from typing import override, Dict, Any

import pandas as pd

from .strategy import Strategy
from .strategy_registry import register_strategy
from src.feature import rate_of_change


@register_strategy("momentum")
class MomentumStrategy(Strategy):
    def __init__(self, simulator, capital, roc_series):
        super().__init__(simulator, capital)
        self.roc = roc_series

    def execute(self, date, price, pred_return, actual_return):
        r = self.roc.get(date)

        if r is None:
            return self.capital, self.entry_price, self.position, self.shares

        if self.position is None and r > 0:
            self.buy(date, price)

        elif self.position == "long" and r < 0:
            self.sell(date, price, pred_return)

        return self.capital, self.entry_price, self.position, self.shares

    @override
    @staticmethod
    def get_extra_params(prices_series: pd.Series) -> Dict[str, Any]:
        roc_period = 12

        roc_values = rate_of_change(prices_series, roc_period)

        return {"roc_series": roc_values.to_dict()}

    @override
    @staticmethod
    def get_minimum_data_points() -> int:
        return 12
