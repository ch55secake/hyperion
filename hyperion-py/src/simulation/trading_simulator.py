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

    def simulate(self,
                predictions,
                actual_returns,
                prices=None,
                dates=None,
                threshold="auto",
                strategy=None,
                use_returns=True,
        ):
        if strategy is None:
            strategy = DirectionalTradingStrategy(self, self.initial_capital, use_returns=use_returns)

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

        # Use the strategy's state - no need to track separately
        for i, pred_return in enumerate(predictions):
            actual_return = actual_returns.iloc[i] if hasattr(actual_returns, "iloc") else actual_returns[i]
            date = dates.iloc[i] if hasattr(dates, "iloc") else (dates[i] if dates is not None else i)

            # Determine current price
            if use_returns:
                current_price = prices.iloc[i] * (1 + actual_return)
            else:
                current_price = prices.iloc[i] if hasattr(prices, "iloc") else prices[i]

            # Execute strategy and let it manage its own state
            strategy.execute(date=date,
                             price=current_price,
                             pred_return=pred_return,
                             actual_return=actual_return)

            # --- Portfolio Tracking ---
            if strategy.use_returns:
                portfolio_value = strategy.capital + (strategy.shares if strategy.position == "long" else 0)
            else:
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
            if strategy.use_returns:
                final_return = actual_returns.iloc[-1] if hasattr(actual_returns, "iloc") else actual_returns[-1]
                strategy.capital = strategy.shares * (1 + final_return)
                final_price = (prices.iloc[-1] if hasattr(prices, "iloc") else prices[-1]) * (1 + final_return)
            else:
                final_price = prices.iloc[-1] if hasattr(prices, "iloc") else prices[-1]
                strategy.capital = strategy.shares * final_price * (1 - self.transaction_cost)

            profit = strategy.capital - self.initial_capital
            pnl = (
                np.mean(actual_returns[-5:]) * 100 if use_returns else ((final_price - strategy.entry_price) / strategy.entry_price) * 100
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

            strategy.shares = 0
            strategy.position = None

        # --- Results ---
        final_value = strategy.capital
        total_return = (final_value - self.initial_capital) / self.initial_capital
        buy_hold_return = None
        if not strategy.use_returns and prices is not None:
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
