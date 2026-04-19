"""Regression tests for the trading simulation layer."""

import numpy as np
import pandas as pd

# Import strategy subpackage so all strategies register themselves.
import src.simulation.strategy  # noqa: F401

from src.simulation.trading_simulator import TradingSimulator
from src.simulation.strategy.strategy_registry import StrategyRegistry


def _make_sim_inputs(n: int = 60, seed: int = 42):
    """Return deterministic (predictions, actual_returns, prices, dates)."""
    rng = np.random.default_rng(seed)
    prices = pd.Series(100.0 + np.cumsum(rng.normal(0, 1, n)))
    actual_returns = prices.pct_change().fillna(0)
    predictions = rng.normal(0, 0.01, n)
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    return predictions, actual_returns, prices, dates


class TestSimulationRegression:
    """Regression tests for the simulation layer using deterministic synthetic data."""

    def test_simulation_returns_deterministic(self):
        """Same predictions + strategy should produce identical P&L on repeated runs."""
        predictions, actual_returns, prices, dates = _make_sim_inputs()

        sim1 = TradingSimulator(initial_capital=10_000)
        result1 = sim1.simulate(predictions, actual_returns, prices=prices, dates=dates)

        sim2 = TradingSimulator(initial_capital=10_000)
        result2 = sim2.simulate(predictions, actual_returns, prices=prices, dates=dates)

        assert (
            result1["final_value"] == result2["final_value"]
        ), "Simulation final value is not deterministic for identical inputs"
        assert result1["num_trades"] == result2["num_trades"], "Trade count is not deterministic for identical inputs"

    def test_all_strategies_run_without_error(self):
        """Every registered strategy should complete simulation without exceptions."""
        n = 60
        predictions, actual_returns, prices, dates = _make_sim_inputs(n=n)
        price_series = prices.copy()
        price_series.index = dates

        available = StrategyRegistry.list()
        assert len(available) > 0, "No strategies registered"

        for strategy_key in available:
            strategy_cls = StrategyRegistry.get(strategy_key)

            min_points = strategy_cls.get_minimum_data_points()
            if n < min_points:
                continue

            extra_params = strategy_cls.get_extra_params(price_series)

            sim = TradingSimulator(initial_capital=10_000)
            strategy = StrategyRegistry.create(
                name=strategy_key,
                simulator=sim,
                capital=10_000,
                **extra_params,
            )

            result = sim.simulate(
                predictions=predictions,
                actual_returns=actual_returns,
                prices=prices,
                dates=dates,
                strategy=strategy,
            )

            assert "final_value" in result, f"Strategy '{strategy_key}' result missing 'final_value'"

    def test_portfolio_value_never_negative(self):
        """No directional strategy should produce negative portfolio values."""
        predictions, actual_returns, prices, dates = _make_sim_inputs()

        sim = TradingSimulator(initial_capital=10_000)
        result = sim.simulate(predictions, actual_returns, prices=prices, dates=dates)

        history = result["portfolio_history"]
        assert (history["portfolio_value"] >= 0).all(), "Portfolio value went negative during simulation"

    def test_final_value_positive_with_always_long_signal(self):
        """When all predictions are positive (always-buy signal), final value should be positive."""
        n = 60
        rng = np.random.default_rng(0)
        prices = pd.Series(100.0 + np.cumsum(np.abs(rng.normal(0.5, 0.3, n))))
        actual_returns = prices.pct_change().fillna(0)
        predictions = np.ones(n) * 0.01  # always positive -> always hold long
        dates = pd.date_range("2023-01-01", periods=n, freq="D")

        sim = TradingSimulator(initial_capital=10_000)
        result = sim.simulate(predictions, actual_returns, prices=prices, dates=dates)
        assert result["final_value"] > 0, "Final portfolio value must be positive"

    def test_portfolio_history_length_equals_input(self):
        """Portfolio history should have one entry per prediction step."""
        n = 40
        predictions, actual_returns, prices, dates = _make_sim_inputs(n=n)
        sim = TradingSimulator(initial_capital=10_000)
        result = sim.simulate(predictions, actual_returns, prices=prices, dates=dates)
        assert len(result["portfolio_history"]) == n
