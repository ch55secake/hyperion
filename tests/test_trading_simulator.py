"""Unit tests for src/simulation/trading_simulator.py"""

import numpy as np
import pandas as pd

from src.simulation.trading_simulator import TradingSimulator
from src.simulation.strategy.adaptive import AdaptiveThresholdStrategy


def _make_inputs(n=50, seed=42):
    """Return deterministic predictions, actual_returns, prices and dates."""
    rng = np.random.default_rng(seed)
    prices = pd.Series(100 + np.cumsum(rng.normal(0, 1, n)))
    actual_returns = prices.pct_change().fillna(0)
    predictions = rng.normal(0, 0.01, n)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    return predictions, actual_returns, prices, dates


class TestTradingSimulatorInit:
    def test_defaults(self):
        sim = TradingSimulator()
        assert sim.initial_capital == 10_000
        assert sim.transaction_cost == 0.001

    def test_custom_capital(self):
        sim = TradingSimulator(initial_capital=50_000)
        assert sim.initial_capital == 50_000


class TestSimulate:
    def test_returns_dict_with_required_keys(self):
        sim = TradingSimulator()
        predictions, actual_returns, prices, dates = _make_inputs()
        result = sim.simulate(predictions, actual_returns, prices=prices, dates=dates)
        assert "final_value" in result
        assert "total_return" in result
        assert "buy_hold_return" in result
        assert "num_trades" in result
        assert "portfolio_history" in result
        assert "trades" in result

    def test_portfolio_history_length(self):
        sim = TradingSimulator()
        predictions, actual_returns, prices, dates = _make_inputs(n=30)
        result = sim.simulate(predictions, actual_returns, prices=prices, dates=dates)
        assert len(result["portfolio_history"]) == 30

    def test_final_value_is_positive(self):
        sim = TradingSimulator()
        predictions, actual_returns, prices, dates = _make_inputs()
        result = sim.simulate(predictions, actual_returns, prices=prices, dates=dates)
        assert result["final_value"] > 0

    def test_buy_hold_return_computed(self):
        sim = TradingSimulator()
        predictions, actual_returns, prices, dates = _make_inputs()
        result = sim.simulate(predictions, actual_returns, prices=prices, dates=dates)
        expected_bh = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]
        assert abs(result["buy_hold_return"] - expected_bh) < 1e-9

    def test_total_return_formula(self):
        sim = TradingSimulator(initial_capital=10_000)
        predictions, actual_returns, prices, dates = _make_inputs()
        result = sim.simulate(predictions, actual_returns, prices=prices, dates=dates)
        expected = (result["final_value"] - 10_000) / 10_000
        assert abs(result["total_return"] - expected) < 1e-9

    def test_auto_threshold(self):
        sim = TradingSimulator()
        predictions, actual_returns, prices, dates = _make_inputs()
        result = sim.simulate(predictions, actual_returns, prices=prices, dates=dates, threshold="auto")
        assert "final_value" in result

    def test_adaptive_threshold(self):
        sim = TradingSimulator()
        predictions, actual_returns, prices, dates = _make_inputs()
        result = sim.simulate(predictions, actual_returns, prices=prices, dates=dates, threshold="adaptive")
        assert "final_value" in result

    def test_fixed_threshold(self):
        sim = TradingSimulator()
        predictions, actual_returns, prices, dates = _make_inputs()
        result = sim.simulate(predictions, actual_returns, prices=prices, dates=dates, threshold=0.005)
        assert "final_value" in result

    def test_portfolio_history_is_dataframe(self):
        sim = TradingSimulator()
        predictions, actual_returns, prices, dates = _make_inputs()
        result = sim.simulate(predictions, actual_returns, prices=prices, dates=dates)
        assert isinstance(result["portfolio_history"], pd.DataFrame)

    def test_trades_is_dataframe(self):
        sim = TradingSimulator()
        predictions, actual_returns, prices, dates = _make_inputs()
        result = sim.simulate(predictions, actual_returns, prices=prices, dates=dates)
        assert isinstance(result["trades"], pd.DataFrame)

    def test_num_trades_matches_trades_dataframe(self):
        sim = TradingSimulator()
        predictions, actual_returns, prices, dates = _make_inputs()
        result = sim.simulate(predictions, actual_returns, prices=prices, dates=dates)
        assert result["num_trades"] == len(result["trades"])


class TestStrategySellGuard:
    def test_sell_before_buy_does_not_raise(self):
        """sell() called before buy() must not raise ZeroDivisionError."""
        from src.simulation.strategy.directional import DirectionalTradingStrategy

        sim = TradingSimulator()
        strategy = DirectionalTradingStrategy(sim, sim.initial_capital)
        # Calling sell when no position is open should be a no-op
        strategy.sell("2024-01-01", 100.0, -0.05)
        assert strategy.position is None
        assert strategy.capital == sim.initial_capital
        assert len(sim.trades) == 0

    def test_sell_with_zero_entry_price_does_not_raise(self):
        """sell() with entry_price == 0 must not raise ZeroDivisionError."""
        from src.simulation.strategy.directional import DirectionalTradingStrategy

        sim = TradingSimulator()
        strategy = DirectionalTradingStrategy(sim, sim.initial_capital)
        # Force a contradictory state: position=long but entry_price=0
        strategy.position = "long"
        strategy.entry_price = 0
        strategy.sell("2024-01-01", 100.0, -0.05)
        # Should return early without appending a trade
        assert len(sim.trades) == 0

    def test_adaptive_strategy_sell_only_when_long(self):
        """AdaptiveThresholdStrategy must not sell when not in a long position."""
        sim = TradingSimulator()
        strategy = AdaptiveThresholdStrategy(sim, sim.initial_capital, threshold=0.01)
        # Trigger sell condition with no open position — should be ignored
        strategy.execute("2024-01-01", 100.0, -0.05, -0.05)
        assert strategy.position is None
        assert strategy.capital == sim.initial_capital
        assert len(sim.trades) == 0
