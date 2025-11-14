from typing import Any, LiteralString

from src.simulation import TradingSimulator
from src.simulation.types import TradeAction, Trade


def execute_threshold_trades(simulator: TradingSimulator,
                             actual_return,
                             capital: int | float | LiteralString | str | Any,
                             current_price: int | Any, date: int | Any, entry_price: int | Any,
                             position: None | str | int | float | Any, pred_return, shares: int | float | Any,
                             threshold,
                             use_returns: bool) -> tuple[
    int | float | LiteralString | str | Any, int | Any, str | int | float | None | Any, int | float | LiteralString | str | Any]:
    if position is None:
        if pred_return > threshold:
            if use_returns:
                shares = capital
                capital = 0
            else:
                shares = (capital * (1 - simulator.transaction_cost)) / current_price
                entry_price = current_price
                capital = 0
            position = "long"
            simulator.trades.append(
                Trade(
                    date=date,
                    action=TradeAction.BUY.value,
                    predicted_return=pred_return,
                    price=current_price,
                    pnl_pct=None,
                    profit=None,
                )
            )
    else:
        if pred_return < -threshold:
            if use_returns:
                capital = shares * (1 + actual_return)
            else:
                capital = shares * current_price * (1 - simulator.transaction_cost)
            profit = capital - simulator.initial_capital
            pnl = (
                actual_return * 100 if use_returns else ((current_price - entry_price) / entry_price) * 100
            )
            simulator.trades.append(
                Trade(
                    date=date,
                    action=TradeAction.SELL.value,
                    predicted_return=pred_return,
                    profit=profit,
                    pnl_pct=pnl,
                    price=current_price,
                )
            )
            shares = 0
            position = None

    return capital, entry_price, position, shares

def simulate_adaptive_threshold_strategy(
        dates_test,
        prices_test,
        predictions,  # ✅ Changed from test_results
        y_test,
) -> tuple[dict, TradingSimulator]:
    print("\n--- Strategy 2: Adaptive Threshold ---")
    print("Uses statistical threshold based on prediction distribution")
    simulator = TradingSimulator(initial_capital=10000)
    adaptive_results = simulator.simulate(
        predictions, y_test, prices_test, dates_test, threshold="adaptive", strategy="threshold"
    )
    return adaptive_results, simulator