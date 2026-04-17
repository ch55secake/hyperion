from typing import override, Dict, Any

import pandas as pd

from .strategy import Strategy
from .strategy_registry import register_strategy
from src.feature import sma


@register_strategy("sma_trend")
class SMATrendStrategy(Strategy):
    def __init__(self, simulator, capital, sma_series):
        super().__init__(simulator, capital)
        self.sma_series = sma_series

    def execute(self, date, price, pred_return, actual_return):
        sma = self.sma_series.get(date)

        if sma is None:
            return self.capital, self.entry_price, self.position, self.shares

        if self.position is None and pred_return > 0 and price > sma:
            self.buy(date, price)

        elif self.position == "long" and price < sma:
            self.sell(date, price, pred_return)

        return self.capital, self.entry_price, self.position, self.shares

    @override
    @staticmethod
    def get_extra_params(prices_series: pd.Series) -> Dict[str, Any]:
        sma_period = 50

        sma_values = sma(prices_series, sma_period)

        return {"sma_series": sma_values.to_dict()}

    @override
    @staticmethod
    def get_minimum_data_points() -> int:
        return 50
