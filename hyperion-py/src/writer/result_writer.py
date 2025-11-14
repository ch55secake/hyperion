from typing import Any
from typing import TYPE_CHECKING

from pandas import DataFrame, Series

from src.xbg import XGBoostStockPredictor

if TYPE_CHECKING:
    from src.simulation import TradingSimulator


def persist_results(
        x,
        x_test: XGBoostStockPredictor,
        x_train: dict[str, float | Any],
        best_strategy: dict[str, DataFrame | float | int | Any],
        period: str,
        sim_results: (
                tuple[str, dict[str, DataFrame | float | int | Any], "TradingSimulator"]
                | dict[str, DataFrame | float | int | Any]
        ),
        strategies: list[tuple[str, dict[str, DataFrame | float | int | Any], "TradingSimulator"]],
        symbol,
        test_results: Series | Any,
        valid_strategies: list[tuple[str, dict[str, DataFrame | float | int | Any], "TradingSimulator"]],
):
    # Step 10: Save detailed results
    results_file = f"results/{symbol}_results.txt"
    with open(results_file, "w", encoding="UTF-8") as f:
        f.write(f"XGBoost Stock Prediction Results for {symbol}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Data Period: {period}\n")
        f.write(f"Total Samples: {len(x)}\n")
        f.write(f"Train Samples: {len(x_train)}\n")
        f.write(f"Test Samples: {len(x_test)}\n\n")
        f.write("Model Performance:\n")
        f.write(f"  Test RMSE: {test_results['rmse']:.8f}\n")
        f.write(f"  Test MAE: {test_results['mae']:.8f}\n")
        f.write(f"  Test R²: {test_results['r2']:.8f}\n\n")

        # Write results for all strategies
        f.write("Trading Simulation Results:\n")
        f.write("  Initial Capital: $10,000.00\n\n")

        for strategy_name, strategy_results, _ in strategies:
            f.write(f"  {strategy_name} Strategy:\n")
            f.write(f"    Final Value: ${strategy_results['final_value']:,.2f}\n")
            f.write(f"    Total Return: {strategy_results['total_return'] * 100:.2f}%\n")
            f.write(f"    Number of Trades: {strategy_results['num_trades']}\n")
            if strategy_results["num_trades"] > 0:
                if strategy_results.get("buy_hold_return") is not None:
                    alpha = (strategy_results["total_return"] - strategy_results["buy_hold_return"]) * 100
                    f.write(f"    Alpha vs Buy&Hold: {alpha:.2f}%\n")
                else:
                    f.write("    Alpha vs Buy&Hold: N/A (no price data)\n")
            f.write("\n")

            if strategy_results.get("buy_hold_return") is not None:
                f.write(f"  Buy & Hold Return: {sim_results['buy_hold_return'] * 100:.2f}%\n")

        if valid_strategies:
            f.write(f"\n  Best Strategy: {best_strategy[0]}\n")
            f.write(f"    Final Value: ${best_strategy[1]['final_value']:,.2f}\n")
            f.write(f"    Total Return: {best_strategy[1]['total_return'] * 100:.2f}%\n")
            f.write(f"    Number of Trades: {best_strategy[1]['num_trades']}\n")

    print(f"✓ Saved detailed results: {results_file}")


def output_best_strategy(
        valid_strategies: list[tuple[str, dict[str, DataFrame | float | int | Any], "TradingSimulator"]],
) -> tuple[
    dict[str, DataFrame | float | int | Any], tuple[str, dict[str, DataFrame | float | int | Any], "TradingSimulator"]
]:
    best_strategy = max(valid_strategies, key=lambda x: (x[1]["total_return"], x[1]["num_trades"]))
    print(f"\n{'=' * 60}")
    print(f"🏆 Best Strategy: {best_strategy[0]}")
    print(f"   Trades: {best_strategy[1]['num_trades']}")
    print(f"   Return: {best_strategy[1]['total_return'] * 100:.2f}%")
    print(f"{'=' * 60}")
    sim_results = best_strategy[1]
    return best_strategy, sim_results
