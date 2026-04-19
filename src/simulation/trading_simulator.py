import numpy as np
import pandas as pd

from src.simulation.risk_metrics import RiskMetrics, compute_risk_metrics
from src.simulation.types import PortfolioHistory, TradeAction, Trade
from src.util import logger
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

        logger.info("=" * 60)
        logger.info("Running Trading Simulation")
        logger.info("=" * 60)

        pred_array = np.asarray(predictions)
        actual_array = np.asarray(actual_returns)
        price_array = np.asarray(prices) if prices is not None else None
        date_array = np.asarray(dates) if dates is not None else None

        if price_array is None:
            raise ValueError("prices must be provided for simulation")

        if threshold == "auto":
            threshold = np.percentile(np.abs(pred_array), 25)
            logger.debug(f"Auto threshold (25th percentile): {threshold:.6f}")
            logger.debug(f"Signals above threshold: {(np.abs(pred_array) > threshold).mean():.1%}")
        elif threshold == "adaptive":
            threshold = 0.3 * np.std(pred_array)
            logger.debug(f"Adaptive threshold (0.3 std): {threshold:.6f}")
        else:
            threshold = float(threshold)
            logger.debug(f"Fixed threshold: {threshold:.6f}")

        # Use the strategy's state - no need to track separately
        for i, pred_return in enumerate(pred_array):
            actual_return = actual_array[i]
            date = date_array[i] if date_array is not None else i

            # Determine current price
            current_price = price_array[i]

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
            final_price = price_array[-1]
            strategy.capital = strategy.shares * final_price * (1 - self.transaction_cost)

            profit = strategy.capital - self.initial_capital
            pnl = ((final_price - strategy.entry_price) / strategy.entry_price) * 100

            final_date = date_array[-1] if date_array is not None else None

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
        if price_array is not None:
            buy_hold_return = (price_array[-1] - price_array[0]) / price_array[0]

        if buy_hold_return is not None and total_return - buy_hold_return > 0:
            logger.info("=" * 60)
            logger.info("Trading Simulation Results")
            logger.info("=" * 60)
            logger.info(f"Initial Capital:       ${self.initial_capital:,.2f}")
            logger.info(f"Final Portfolio Value: ${final_value:,.2f}")
            logger.info(f"Total Return:          {total_return * 100:.2f}%")
            if buy_hold_return is not None:
                logger.info(f"Buy & Hold Return:     {buy_hold_return * 100:.2f}%")
                logger.info(f"Strategy Alpha:        {(total_return - buy_hold_return) * 100:.2f}%")
            logger.info(f"Number of Trades:      {len(self.trades)}")

        portfolio_df = pd.DataFrame(self.portfolio_history)
        trades_df = pd.DataFrame(self.trades)
        risk_metrics: RiskMetrics = compute_risk_metrics(
            portfolio_history=portfolio_df,
            trades=trades_df,
            total_return=total_return,
        )

        return {
            "portfolio_history": portfolio_df,
            "trades": trades_df,
            "final_value": final_value,
            "total_return": total_return,
            "buy_hold_return": buy_hold_return,
            "num_trades": len(self.trades),
            "risk_metrics": risk_metrics,
        }
