"""Unit tests for src/simulation/strategy/regime_aware.py"""

import numpy as np

import src.simulation.strategy  # noqa: F401 — trigger all @register_strategy decorators
from src.feature.regime import REGIME_BEAR, REGIME_BULL, REGIME_SIDEWAYS
from src.simulation.strategy.regime_aware import RegimeAwareStrategy
from src.simulation.strategy.strategy_registry import StrategyRegistry
from src.simulation.trading_simulator import TradingSimulator
from tests.helpers import make_ohlcv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_simulator(capital: float = 10_000) -> TradingSimulator:
    return TradingSimulator(initial_capital=capital, transaction_cost=0.001)


def _make_strategy(regime_series: dict, capital: float = 10_000) -> tuple[RegimeAwareStrategy, TradingSimulator]:
    sim = _make_simulator(capital)
    strategy = RegimeAwareStrategy(sim, capital, regime_series=regime_series)
    return strategy, sim


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestRegimeAwareStrategyRegistration:
    def test_is_registered(self):
        assert "regime_aware" in StrategyRegistry.list()

    def test_can_be_created_via_registry(self):
        sim = _make_simulator()
        strategy = StrategyRegistry.create("regime_aware", sim, 10_000, regime_series={})
        assert isinstance(strategy, RegimeAwareStrategy)


# ---------------------------------------------------------------------------
# execute: bull regime (momentum)
# ---------------------------------------------------------------------------


class TestBullRegimeMomentum:
    def test_buy_on_positive_prediction_in_bull(self):
        regime_series = {"2023-01-01": REGIME_BULL}
        strategy, sim = _make_strategy(regime_series)
        strategy.execute("2023-01-01", price=100.0, pred_return=0.02, actual_return=0.01)
        assert strategy.position == "long"

    def test_no_buy_on_negative_prediction_in_bull(self):
        regime_series = {"2023-01-01": REGIME_BULL}
        strategy, sim = _make_strategy(regime_series)
        strategy.execute("2023-01-01", price=100.0, pred_return=-0.01, actual_return=-0.005)
        assert strategy.position is None

    def test_sell_on_negative_prediction_in_bull(self):
        regime_series = {
            "2023-01-01": REGIME_BULL,
            "2023-01-02": REGIME_BULL,
        }
        strategy, sim = _make_strategy(regime_series)
        # Enter
        strategy.execute("2023-01-01", price=100.0, pred_return=0.02, actual_return=0.01)
        assert strategy.position == "long"
        # Exit
        strategy.execute("2023-01-02", price=102.0, pred_return=-0.01, actual_return=0.0)
        assert strategy.position is None


# ---------------------------------------------------------------------------
# execute: bear regime (conservative)
# ---------------------------------------------------------------------------


class TestBearRegimeConservative:
    def test_no_buy_in_bear_regime(self):
        regime_series = {"2023-01-01": REGIME_BEAR}
        strategy, sim = _make_strategy(regime_series)
        strategy.execute("2023-01-01", price=100.0, pred_return=0.05, actual_return=0.02)
        assert strategy.position is None

    def test_sell_long_on_negative_prediction_in_bear(self):
        regime_series = {
            "2023-01-01": REGIME_BULL,  # enter in bull
            "2023-01-02": REGIME_BEAR,  # exit in bear
        }
        strategy, sim = _make_strategy(regime_series)
        # Enter position in bull
        strategy.execute("2023-01-01", price=100.0, pred_return=0.02, actual_return=0.01)
        assert strategy.position == "long"
        # Bear regime + negative prediction → close
        strategy.execute("2023-01-02", price=95.0, pred_return=-0.02, actual_return=-0.01)
        assert strategy.position is None


# ---------------------------------------------------------------------------
# execute: sideways regime (mean reversion)
# ---------------------------------------------------------------------------


class TestSidewaysRegimeReversion:
    def test_buy_above_reversion_threshold(self):
        regime_series = {"2023-01-01": REGIME_SIDEWAYS}
        strategy, sim = _make_strategy(regime_series)
        strategy.execute("2023-01-01", price=100.0, pred_return=0.01, actual_return=0.005)
        assert strategy.position == "long"

    def test_no_buy_below_reversion_threshold(self):
        regime_series = {"2023-01-01": REGIME_SIDEWAYS}
        strategy, sim = _make_strategy(regime_series)
        # Default reversion threshold is 0.005, so 0.003 should not trigger a buy
        strategy.execute("2023-01-01", price=100.0, pred_return=0.003, actual_return=0.002)
        assert strategy.position is None

    def test_sell_below_negative_reversion_threshold(self):
        regime_series = {
            "2023-01-01": REGIME_SIDEWAYS,
            "2023-01-02": REGIME_SIDEWAYS,
        }
        strategy, sim = _make_strategy(regime_series)
        # Enter
        strategy.execute("2023-01-01", price=100.0, pred_return=0.01, actual_return=0.01)
        assert strategy.position == "long"
        # Exit
        strategy.execute("2023-01-02", price=99.0, pred_return=-0.01, actual_return=-0.01)
        assert strategy.position is None


# ---------------------------------------------------------------------------
# execute: fallback when date not in regime_series
# ---------------------------------------------------------------------------


class TestRegimeFallback:
    def test_unknown_date_defaults_to_sideways(self):
        """An unrecognised date should fall back to sideways (mean-reversion) logic."""
        strategy, sim = _make_strategy(regime_series={})
        # Sideways entry threshold = 0.005; pred=0.01 > threshold → should buy
        strategy.execute("2099-01-01", price=100.0, pred_return=0.01, actual_return=0.0)
        assert strategy.position == "long"


# ---------------------------------------------------------------------------
# execute: return tuple structure
# ---------------------------------------------------------------------------


class TestExecuteReturnTuple:
    def test_returns_four_element_tuple(self):
        strategy, sim = _make_strategy(regime_series={"d": REGIME_BULL})
        result = strategy.execute("d", price=100.0, pred_return=0.02, actual_return=0.01)
        assert len(result) == 4

    def test_capital_position_entry_shares_consistent(self):
        strategy, sim = _make_strategy(regime_series={"d": REGIME_BULL})
        capital, entry, position, shares = strategy.execute("d", price=100.0, pred_return=0.02, actual_return=0.0)
        assert capital == strategy.capital
        assert position == strategy.position
        assert shares == strategy.shares


# ---------------------------------------------------------------------------
# get_extra_params
# ---------------------------------------------------------------------------


class TestGetExtraParams:
    def test_returns_regime_series_key(self):
        prices = make_ohlcv(n=200, seed=42)["Close"]
        params = RegimeAwareStrategy.get_extra_params(prices)
        assert "regime_series" in params

    def test_regime_series_keys_match_price_index(self):
        ohlcv = make_ohlcv(n=200, seed=42)
        prices = ohlcv["Close"]
        params = RegimeAwareStrategy.get_extra_params(prices)
        regime_keys = set(params["regime_series"].keys())
        price_keys = set(prices.index)
        assert regime_keys == price_keys

    def test_regime_values_are_integers(self):
        prices = make_ohlcv(n=200, seed=42)["Close"]
        params = RegimeAwareStrategy.get_extra_params(prices)
        for v in params["regime_series"].values():
            assert isinstance(v, (int, np.integer)), f"Expected int, got {type(v)}"


# ---------------------------------------------------------------------------
# regime_name utility
# ---------------------------------------------------------------------------


class TestRegimeName:
    def test_known_labels(self):
        assert RegimeAwareStrategy.regime_name(REGIME_BULL) == "bull"
        assert RegimeAwareStrategy.regime_name(REGIME_BEAR) == "bear"
        assert RegimeAwareStrategy.regime_name(REGIME_SIDEWAYS) == "sideways"

    def test_unknown_label_returns_unknown(self):
        assert RegimeAwareStrategy.regime_name(99) == "unknown"


# ---------------------------------------------------------------------------
# get_minimum_data_points
# ---------------------------------------------------------------------------


class TestGetMinimumDataPoints:
    def test_returns_integer(self):
        assert isinstance(RegimeAwareStrategy.get_minimum_data_points(), int)

    def test_minimum_is_reasonable(self):
        assert RegimeAwareStrategy.get_minimum_data_points() >= 10


# ---------------------------------------------------------------------------
# End-to-end: full simulation loop
# ---------------------------------------------------------------------------


class TestRegimeAwareEndToEnd:
    def test_simulation_runs_without_error(self):
        """Full simulate() call should complete and return a result dict."""
        from src.simulation.strategy.strategy import Strategy

        ohlcv = make_ohlcv(n=100, seed=42)
        prices = ohlcv["Close"]
        returns = prices.pct_change(1).fillna(0)

        from src.feature.regime import classify_regime

        regime_df = classify_regime(prices, returns)
        regime_series = regime_df["Regime_Trend"].to_dict()

        preds = returns.values + np.random.default_rng(0).normal(0, 0.005, len(returns))
        dates = prices.index

        sim = _make_simulator()
        strategy = RegimeAwareStrategy(sim, sim.initial_capital, regime_series=regime_series)
        result, _ = Strategy.simulate(strategy, dates, prices.values, preds, returns.values)

        assert "total_return" in result
        assert "num_trades" in result
        assert isinstance(result["num_trades"], int)
