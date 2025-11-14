from typing import Any, LiteralString

from src.simulation import TradingSimulator
from src.simulation.types import Threshold, Trade, TradeAction


def execute_hold_days_trades(simulator: TradingSimulator, actual_return,
                             capital: int | float | LiteralString | str | Any,
                             current_price: int | Any, date: int | Any, entry_price: int | Any, hold_counter: int,
                             position: None | str | int | float | Any, pred_return,
                             shares: int | float | LiteralString | str | Any,
                             threshold: Threshold, use_returns: bool) -> tuple[
    int | float | LiteralString | Any, int | Any, Any, int]:
    if position is None and pred_return > threshold:
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
    elif position == "long":
        hold_counter += 1
        if hold_counter >= 5 or pred_return < -threshold:
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


def simulate_hold_days_strategy(
        dates_test,
        prices_test,
        predictions,  # ✅ Changed from test_results
        y_test,
) -> tuple[dict, TradingSimulator]:
    print("\n--- Strategy 3: Hold Days Strategy ---")
    print("Holds positions for multiple days")
    simulator = TradingSimulator(initial_capital=10000)
    hold_days_results = simulator.simulate(
        predictions, y_test, prices_test, dates_test, threshold="adaptive", strategy="hold_days"
    )
    return hold_days_results, simulator
