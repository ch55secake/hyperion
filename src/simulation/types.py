from dataclasses import dataclass
from enum import Enum
from typing import Any


class TradeAction(Enum):
    BUY = "BUY"
    SELL = "SELL"
    SELL_FINAL = "SELL (Final)"
    # HOLD = "HOLD"


@dataclass
class Trade:
    date: Any
    action: str  # TradeAction
    predicted_return: float | None
    profit: float | None
    pnl_pct: float | None
    price: float


@dataclass
class PortfolioHistory:
    date: Any
    portfolio_value: float
    position: str
    prediction: float
    price: float
