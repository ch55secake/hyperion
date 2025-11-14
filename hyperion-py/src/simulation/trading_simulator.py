from typing import LiteralString, Any

import numpy as np
import pandas as pd

from src.simulation.types import PortfolioHistory, TradeAction, Trade


class TradingSimulator:
    """
    Simulates trading based on model predictions
    """

    def __init__(self, initial_capital=10000, transaction_cost=0.001):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.portfolio_history = []
        self.trades = []

    def simulate(
            self,
            predictions,
            actual_returns,
            prices=None,
            dates=None,
            threshold="auto",
            strategy="directional",
            use_returns=True,
    ):
        """
        Simulate trading strategy and evaluate predictive performance.
        """
        print("\n" + "=" * 60)
        print("Running Trading Simulation")
        print("=" * 60)

        pred_array = np.array(predictions)

        # --- Determine Threshold ---
        if threshold == "auto":
            threshold = np.percentile(np.abs(pred_array), 25)
            print(f"\nAuto threshold (25th percentile): {threshold:.6f}")
        elif threshold == "adaptive":
            threshold = 0.3 * np.std(pred_array)
            print(f"\nAdaptive threshold (0.3 std): {threshold:.6f}")
        else:
            threshold = float(threshold)
            print(f"\nFixed threshold: {threshold:.6f}")

        # --- Initialize Portfolio ---
        capital = self.initial_capital
        shares = 0
        position = None
        entry_price = 0
        hold_counter = 0

        for i, pred_return in enumerate(predictions):
            actual_return = actual_returns.iloc[i] if hasattr(actual_returns, "iloc") else actual_returns[i]
            date = dates.iloc[i] if hasattr(dates, "iloc") else (dates[i] if dates is not None else i)

            # Determine current price
            if use_returns:
                current_price = prices.iloc[i] * (1 + actual_return)
            else:
                current_price = prices.iloc[i] if hasattr(prices, "iloc") else prices[i]

            if strategy == "directional":
                capital, entry_price, position, shares = self.execute_directional_trades(
                                                                                actual_return=actual_return,
                                                                                capital=capital,
                                                                                current_price=current_price,
                                                                                date=date,
                                                                                entry_price=entry_price,
                                                                                position=position,
                                                                                pred_return=pred_return,
                                                                                shares=shares,
                                                                                use_returns=use_returns)
            elif strategy == "threshold":
                capital, entry_price, position, shares = self.execute_threshold_trades(
                                                                              actual_return=actual_return,
                                                                              capital=capital,
                                                                              current_price=current_price,
                                                                              date=date,
                                                                              entry_price=entry_price,
                                                                              position=position,
                                                                              pred_return=pred_return,
                                                                              shares=shares,
                                                                              threshold=threshold,
                                                                              use_returns=use_returns)
            elif strategy == "hold_days":
                capital, entry_price, position, shares = self.execute_hold_days_trades(actual_return=actual_return,
                                                                              capital=capital,
                                                                              current_price=current_price,
                                                                              date=date,
                                                                              entry_price=entry_price,
                                                                              position=position,
                                                                              pred_return=pred_return,
                                                                              threshold=threshold,
                                                                              hold_counter=hold_counter,
                                                                              shares=shares,
                                                                              use_returns=use_returns)

            # --- Portfolio Tracking ---
            if use_returns:
                portfolio_value = capital + (shares if position == "long" else 0)
            else:
                portfolio_value = shares * current_price if position == "long" else capital
            self.portfolio_history.append(
                PortfolioHistory(
                    date=date,
                    portfolio_value=portfolio_value,
                    position=position if position else "cash",
                    prediction=pred_return,
                    price=current_price,
                )
            )

        # --- Close remaining position ---
        if position is not None:
            if use_returns:
                final_return = actual_returns.iloc[-1] if hasattr(actual_returns, "iloc") else actual_returns[-1]
                capital = shares * (1 + final_return)
                final_price = (prices.iloc[-1] if hasattr(prices, "iloc") else prices[-1]) * (1 + final_return)
            else:
                final_price = prices.iloc[-1] if hasattr(prices, "iloc") else prices[-1]
                capital = shares * final_price * (1 - self.transaction_cost)

            profit = capital - self.initial_capital
            pnl = (
                np.mean(actual_returns[-5:]) * 100 if use_returns else ((final_price - entry_price) / entry_price) * 100
            )

            if hasattr(dates, "__getitem__"):
                final_date = dates[-1]
            else:
                final_date = None

            self.trades.append(
                Trade(
                    date=final_date,
                    action=TradeAction.SELL_FINAL.value,
                    profit=profit,
                    pnl_pct=pnl,
                    price=final_price,
                    predicted_return=None,
                )
            )

            shares = 0

        # --- Results ---
        final_value = capital
        total_return = (final_value - self.initial_capital) / self.initial_capital
        buy_hold_return = None
        if not use_returns and prices is not None:
            first_price, last_price = prices.iloc[0], prices.iloc[-1]
            buy_hold_return = (last_price - first_price) / first_price

        print("\n" + "=" * 60)
        print("Trading Simulation Results")
        print("=" * 60)
        print(f"Initial Capital:       ${self.initial_capital:,.2f}")
        print(f"Final Portfolio Value: ${final_value:,.2f}")
        print(f"Total Return:          {total_return * 100:.2f}%")
        if buy_hold_return is not None:
            print(f"Buy & Hold Return:     {buy_hold_return * 100:.2f}%")
            print(f"Strategy Alpha:        {(total_return - buy_hold_return) * 100:.2f}%")
        print(f"Number of Trades:      {len(self.trades)}")

        return {
            "portfolio_history": pd.DataFrame(self.portfolio_history),
            "trades": pd.DataFrame(self.trades),
            "final_value": final_value,
            "total_return": total_return,
            "buy_hold_return": buy_hold_return,
            "num_trades": len(self.trades),
        }


    def execute_hold_days_trades(self, actual_return,
                                 capital: int | float | LiteralString | str | Any,
                                 current_price: int | Any, date: int | Any, entry_price: int | Any, hold_counter: int,
                                 position: None | str | int | float | Any, pred_return,
                                 shares: int | float | LiteralString | str | Any,
                                 threshold, use_returns: bool) -> tuple[
        int | float | LiteralString | Any, int | Any, Any, int]:
        if position is None and pred_return > threshold:
            if use_returns:
                shares = capital
                capital = 0
            else:
                shares = (capital * (1 - self.transaction_cost)) / current_price
                entry_price = current_price
                capital = 0
            position = "long"
            self.trades.append(
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
                    capital = shares * current_price * (1 - self.transaction_cost)
                profit = capital - self.initial_capital
                pnl = (
                    actual_return * 100 if use_returns else ((current_price - entry_price) / entry_price) * 100
                )
                self.trades.append(
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

    def execute_threshold_trades(self,
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
                    shares = (capital * (1 - self.transaction_cost)) / current_price
                    entry_price = current_price
                    capital = 0
                position = "long"
                self.trades.append(
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
                    capital = shares * current_price * (1 - self.transaction_cost)
                profit = capital - self.initial_capital
                pnl = (
                    actual_return * 100 if use_returns else ((current_price - entry_price) / entry_price) * 100
                )
                self.trades.append(
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


    def execute_directional_trades(self, actual_return, capital: int | float | Any, current_price: int | Any,
                                   date: int | Any, entry_price: int | Any, position: str | None, pred_return,
                                   shares: int | float | Any,
                                   use_returns: bool) -> tuple[str, int, int | float | Any, int | Any]:
        if position is None and pred_return > 0:
            # Buy
            if use_returns:
                shares = capital
                capital = 0
            else:
                shares = (capital * (1 - self.transaction_cost)) / current_price
                entry_price = current_price
                capital = 0
            position = "long"
            self.trades.append(
                Trade(
                    date=date,
                    action=TradeAction.BUY.value,
                    predicted_return=pred_return,
                    price=current_price,
                    pnl_pct=None,
                    profit=None,
                )
            )

        elif position == "long" and pred_return <= 0:
            # Sell
            if use_returns:
                capital = shares * (1 + actual_return)
            else:
                capital = shares * current_price * (1 - self.transaction_cost)
            profit = capital - self.initial_capital
            pnl = actual_return * 100 if use_returns else ((current_price - entry_price) / entry_price) * 100
            self.trades.append(
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

def simulate_directional_trading_strategy(
        dates_test,
        prices_test,
        predictions,  # ✅ Changed from test_results
        y_test,
) -> tuple[dict, TradingSimulator]:
    print("\n--- Strategy 1: Directional Trading ---")
    print("Buys when prediction > 0, sells when prediction <= 0")
    simulator = TradingSimulator(initial_capital=10000)
    sim_results = simulator.simulate(
        predictions,  # ✅ Use predictions directly
        y_test,
        prices_test,
        dates_test,
        threshold=0,
        strategy="directional",
    )
    return sim_results, simulator


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
