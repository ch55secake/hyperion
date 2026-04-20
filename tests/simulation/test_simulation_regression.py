"""Regression tests for the trading simulation layer."""

import numpy as np
import pandas as pd
import pytest

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


def _make_foresight_inputs(n: int = 120, seed: int = 0):
    """Return (predictions, actual_returns, prices, dates, price_series) for a strongly
    trending synthetic series where predictions == actual_returns (perfect foresight).

    Daily log-returns average ~3 %, well above the 0.02 threshold used by adaptive /
    hold_days strategies, so every prediction-driven strategy will trade.
    """
    rng = np.random.default_rng(seed)
    log_returns = 0.03 + rng.normal(0, 0.005, n)
    prices = pd.Series(100.0 * np.exp(np.cumsum(log_returns)))
    actual_returns = prices.pct_change().fillna(float(np.exp(log_returns[0]) - 1))
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    price_series = prices.copy()
    price_series.index = dates
    predictions = actual_returns.to_numpy()
    return predictions, actual_returns, prices, dates, price_series


# Strategies that do not use the prediction signal for entry/exit decisions,
# are non-deterministic (coinflip), or are explicitly designed to bet against
# the signal (contrarian).  These are excluded from signal-quality checks.
_NON_SIGNAL_STRATEGIES: frozenset[str] = frozenset({"coinflip", "contrarian", "momentum", "bb_reversion", "ema_cross"})

_SIGNAL_DRIVEN_STRATEGIES: list[str] = [k for k in StrategyRegistry.list() if k not in _NON_SIGNAL_STRATEGIES]


class TestSimulationRegression:
    """Regression tests for the simulation layer using deterministic synthetic data."""

    def test_simulation_returns_deterministic(self):
        """Same predictions + strategy should produce identical P&L on repeated runs."""
        predictions, actual_returns, prices, dates = _make_sim_inputs()

        sim1 = TradingSimulator(initial_capital=10_000)
        result1 = sim1.simulate(predictions, actual_returns, prices=prices, dates=dates)

        sim2 = TradingSimulator(initial_capital=10_000)
        result2 = sim2.simulate(predictions, actual_returns, prices=prices, dates=dates)

        assert result1["final_value"] == result2["final_value"], (
            "Simulation final value is not deterministic for identical inputs"
        )
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


class TestPerfectForesight:
    """Economic-correctness smoke tests using perfect-foresight and zero-signal inputs."""

    @pytest.mark.parametrize("strategy_key", _SIGNAL_DRIVEN_STRATEGIES)
    def test_perfect_foresight_positive_return(self, strategy_key):
        """Passing actual future returns as predictions should yield a positive total
        return for every prediction-driven strategy (after transaction costs).

        A negative return with perfect information indicates fundamentally broken
        buy/sell logic — e.g. inverted entry conditions or miscalculated sizing.
        """
        n = 120
        predictions, actual_returns, prices, dates, price_series = _make_foresight_inputs(n=n)

        strategy_cls = StrategyRegistry.get(strategy_key)
        if n < strategy_cls.get_minimum_data_points():
            pytest.skip(f"Not enough synthetic data points for '{strategy_key}'")

        extra_params = strategy_cls.get_extra_params(price_series)
        sim = TradingSimulator(initial_capital=10_000, transaction_cost=0.001)
        strategy = StrategyRegistry.create(name=strategy_key, simulator=sim, capital=10_000, **extra_params)

        result = sim.simulate(
            predictions=predictions,
            actual_returns=actual_returns,
            prices=prices,
            dates=dates,
            strategy=strategy,
        )

        assert result["total_return"] > 0, (
            f"Strategy '{strategy_key}' produced a non-positive return "
            f"({result['total_return']:.4%}) with perfect-foresight predictions on a "
            "strongly trending series — buy/sell logic may be inverted or broken."
        )

    @pytest.mark.parametrize("strategy_key", _SIGNAL_DRIVEN_STRATEGIES)
    def test_zero_signal_no_manufactured_return(self, strategy_key):
        """With constant-zero predictions, no prediction-driven strategy should open
        any position or manufacture returns from nothing.

        A non-zero return here indicates the strategy is ignoring its prediction
        input, or that entry logic fires even when the signal is absent.
        """
        n = 120
        _, actual_returns, prices, dates, price_series = _make_foresight_inputs(n=n)
        predictions = np.zeros(n)

        strategy_cls = StrategyRegistry.get(strategy_key)
        if n < strategy_cls.get_minimum_data_points():
            pytest.skip(f"Not enough synthetic data points for '{strategy_key}'")

        extra_params = strategy_cls.get_extra_params(price_series)
        sim = TradingSimulator(initial_capital=10_000, transaction_cost=0.001)
        strategy = StrategyRegistry.create(name=strategy_key, simulator=sim, capital=10_000, **extra_params)

        result = sim.simulate(
            predictions=predictions,
            actual_returns=actual_returns,
            prices=prices,
            dates=dates,
            strategy=strategy,
        )

        assert result["num_trades"] == 0, (
            f"Strategy '{strategy_key}' made {result['num_trades']} trade(s) on a "
            "zero-prediction signal — it may be ignoring the prediction input."
        )
        assert result["total_return"] == 0.0, (
            f"Strategy '{strategy_key}' produced a non-zero return "
            f"({result['total_return']:.4%}) from a zero-prediction signal."
        )
