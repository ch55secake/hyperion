from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from numpy._typing import NDArray

Numeric = np.floating[Any] | np.complexfloating[Any, Any] | float
Date = np.timedelta64 | np.datetime64
Array = NDArray[Any]

Threshold = Numeric | Date | Array | Any

class TradeAction(Enum):
    BUY = "BUY"
    SELL = "SELL"
    SELL_FINAL = "SELL (Final)"
    # HOLD = "HOLD"


@dataclass
class Trade:
    date: str
    action: str  # TradeAction
    predicted_return: float | None
    profit: float | None
    pnl_pct: float | None
    price: float


@dataclass
class PortfolioHistory:
    date: str
    portfolio_value: float
    position: str
    prediction: float
    price: float

