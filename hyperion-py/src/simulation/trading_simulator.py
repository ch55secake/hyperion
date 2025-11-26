from dataclasses import dataclass
from typing import LiteralString, Any

import numpy as np
import pandas as pd

from src.simulation.types import PortfolioHistory, TradeAction, Trade
from .strategy.directional import DirectionalTradingStrategy


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
        strategy=None,
    ):
        """
        Simulate trading strategy and evaluate predictive performance.
        """
        if strategy is None:
            strategy = DirectionalTradingStrategy(self, self.initial_capital)

        print("\n" + "=" * 60)
        print("Running Trading Simulation")
        print("=" * 60)

        pred_array = np.array(predictions)

        if threshold == "auto":
            threshold = np.percentile(np.abs(pred_array), 25)
            print(f"\nAuto threshold (25th percentile): {threshold:.6f}")
        elif threshold == "adaptive":
            threshold = 0.3 * np.std(pred_array)
            print(f"\nAdaptive threshold (0.3 std): {threshold:.6f}")
        else:
            threshold = float(threshold)
            print(f"\nFixed threshold: {threshold:.6f}")

        # Use the strategy's state - no need to track separately
        for i, pred_return in enumerate(predictions):
            actual_return = actual_returns.iloc[i] if hasattr(actual_returns, "iloc") else actual_returns[i]
            date = dates.iloc[i] if hasattr(dates, "iloc") else (dates[i] if dates is not None else i)

            # Determine current price
            current_price = prices.iloc[i] if hasattr(prices, "iloc") else prices[i]

            # Execute strategy and let it manage its own state
            strategy.execute(date=date, price=current_price, pred_return=pred_return, actual_return=actual_return)

            # --- Portfolio Tracking ---
            portfolio_value = strategy.shares * current_price if strategy.position == "long" else strategy.capital
            self.portfolio_history.append(
                PortfolioHistory(
                    date=date,
                    portfolio_value=portfolio_value,
                    position=strategy.position if strategy.position else "cash",
                    prediction=pred_return,
                    price=current_price,
                )
            )

        # --- Close remaining position ---
        if strategy.position is not None:
            final_price = prices.iloc[-1] if hasattr(prices, "iloc") else prices[-1]
            strategy.capital = strategy.shares * final_price * (1 - self.transaction_cost)

            profit = strategy.capital - self.initial_capital
            pnl = ((final_price - strategy.entry_price) / strategy.entry_price) * 100

            # Get final date using iloc for pandas Series
            if hasattr(dates, "iloc"):
                final_date = dates.iloc[-1]
            elif hasattr(dates, "__getitem__"):
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

            strategy.shares = 0
            strategy.position = None

        # --- Results ---
        final_value = strategy.capital
        total_return = (final_value - self.initial_capital) / self.initial_capital
        buy_hold_return = None
        if prices is not None:
            first_price = prices.iloc[0] if hasattr(prices, "iloc") else prices[0]
            last_price = prices.iloc[-1] if hasattr(prices, "iloc") else prices[-1]
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
