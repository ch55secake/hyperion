from typing import Any

from .strategy import Strategy

class DirectionalTradingStrategy(Strategy):
    def __init__(self, simulator: Any, capital: Any, use_returns:bool =True):
        super().__init__(simulator, capital, use_returns)

    def execute(self,
                date: int | Any,
                price: int | Any,
                pred_return: Any,
                actual_return: Any) -> tuple[int | float | Any, int | Any, Any, int | Any]:

        if self.position is None and pred_return > 0:
            self.buy(date, price, pred_return)

        elif self.position == "long" and pred_return <= 0:
            self.sell(date, price, pred_return, actual_return)

        return self.capital, self.entry_price, self.position, self.shares


# def execute_directional_trades(simulator: TradingSimulator,
#                                actual_return,
#                                capital: int | float | Any,
#                                current_price: int | Any,
#                                date: int | Any,
#                                entry_price: int | Any,
#                                position: str | None,
#                                pred_return,
#                                shares: int | float | Any,
#                                use_returns: bool) -> tuple[str, int, int | float | Any, int | Any]:
#     if position is None and pred_return > 0:
#         # Buy
#         if use_returns:
#             shares = capital
#             capital = 0
#         else:
#             shares = (capital * (1 - simulator.transaction_cost)) / current_price
#             entry_price = current_price
#             capital = 0
#         position = "long"
#         simulator.trades.append(
#             Trade(
#                 date=date,
#                 action=TradeAction.BUY.value,
#                 predicted_return=pred_return,
#                 price=current_price,
#                 pnl_pct=None,
#                 profit=None,
#             )
#         )
#
#     elif position == "long" and pred_return <= 0:
#         # Sell
#         if use_returns:
#             capital = shares * (1 + actual_return)
#         else:
#             capital = shares * current_price * (1 - simulator.transaction_cost)
#         profit = capital - simulator.initial_capital
#         pnl = actual_return * 100 if use_returns else ((current_price - entry_price) / entry_price) * 100
#         simulator.trades.append(
#             Trade(
#                 date=date,
#                 action=TradeAction.SELL.value,
#                 predicted_return=pred_return,
#                 profit=profit,
#                 pnl_pct=pnl,
#                 price=current_price,
#             )
#         )
#         shares = 0
#         position = None
#
#     return capital, entry_price, position, shares
#
# def simulate_directional_trading_strategy(
#         dates_test,
#         prices_test,
#         predictions,  # ✅ Changed from test_results
#         y_test,
# ) -> tuple[dict, TradingSimulator]:
#     print("\n--- Strategy 1: Directional Trading ---")
#     print("Buys when prediction > 0, sells when prediction <= 0")
#     simulator = TradingSimulator(initial_capital=10000)
#     sim_results = simulator.simulate(
#         predictions,  # ✅ Use predictions directly
#         y_test,
#         prices_test,
#         dates_test,
#         threshold=0,
#         strategy="directional",
#     )
#     return sim_results, simulator
