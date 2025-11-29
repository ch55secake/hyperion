from typing import override, Dict, Any

import pandas as pd

from .strategy import Strategy
from .strategy_registry import register_strategy
from src.feature import sma


@register_strategy("hybrid_trend_ml")
class HybridTrendMLStrategy(Strategy):
    def __init__(self, simulator, capital, sma_series, threshold=0.01):
        super().__init__(simulator, capital)
        self.sma = sma_series
        self.threshold = threshold

    def execute(self, date, price, pred_return, actual_return):
        sma = self.sma.get(date)

        if sma is None:
            return self.capital, self.entry_price, self.position, self.shares

        if self.position is None and pred_return > self.threshold and price > sma:
            self.buy(date, price)

        elif self.position == "long" and (pred_return < -self.threshold or price < sma):
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
