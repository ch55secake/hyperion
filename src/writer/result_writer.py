from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

from pandas import DataFrame, Series

from src.util import logger

if TYPE_CHECKING:
    from src.simulation.trading_simulator import TradingSimulator


def persist_results(
    x: DataFrame,
    x_test: Dict[str, DataFrame],
    x_train: Dict[str, DataFrame],
    best_strategy: dict[str, DataFrame | float | int | Any],
    period: str,
    sim_results: (
        tuple[str, dict[str, DataFrame | float | int | Any], "TradingSimulator"]
        | dict[str, DataFrame | float | int | Any]
    ),
    strategies: list[tuple[str, dict[str, DataFrame | float | int | Any], "TradingSimulator"]],
    symbol: str,
    test_results: Series | Any,
    valid_strategies: list[tuple[str, dict[str, DataFrame | float | int | Any], "TradingSimulator"]],
):
    # Step 10: Save detailed results
    results_file = f"results/{symbol}_results.txt"
    with open(results_file, "w", encoding="UTF-8") as f:
        f.write(f"Stock Prediction Results for {symbol}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Data Period: {period}\n")
        f.write(f"Total Samples: {len(x)}\n")
        train_samples = (
            len(x_train.get("daily", next(iter(x_train.values()))))
            if isinstance(x_train, dict) and x_train
            else len(x_train)
        )
        test_samples = (
            len(x_test.get("daily", next(iter(x_test.values()))))
            if isinstance(x_test, dict) and x_test
            else len(x_test)
        )
        f.write(f"Train Samples: {train_samples}\n")
        f.write(f"Test Samples: {test_samples}\n\n")
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
                f.write(f"  Buy & Hold Return: {strategy_results['buy_hold_return'] * 100:.2f}%\n")

        if valid_strategies:
            f.write(f"\n  Best Strategy: {best_strategy[0]}\n")
            f.write(f"    Final Value: ${best_strategy[1]['final_value']:,.2f}\n")
            f.write(f"    Total Return: {best_strategy[1]['total_return'] * 100:.2f}%\n")
            f.write(f"    Number of Trades: {best_strategy[1]['num_trades']}\n")

    logger.info(f"Saved detailed results: {results_file}")


def output_best_strategy(
    valid_strategies: list[tuple[str, dict[str, DataFrame | float | int | Any], "TradingSimulator"]],
) -> tuple[
    dict[str, DataFrame | float | int | Any], tuple[str, dict[str, DataFrame | float | int | Any], "TradingSimulator"]
]:
    best_strategy = max(valid_strategies, key=lambda x: (x[1]["total_return"], x[1]["num_trades"]))
    logger.info("=" * 60)
    logger.info(f"Best Strategy: {best_strategy[0]}")
    logger.info(f"   Trades: {best_strategy[1]['num_trades']}")
    logger.info(f"   Return: {best_strategy[1]['total_return'] * 100:.2f}%")
    logger.info("=" * 60)
    sim_results = best_strategy[1]
    return best_strategy, sim_results
