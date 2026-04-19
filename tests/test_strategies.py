"""Unit tests for all trading strategies in src/simulation/strategy/."""

import random

import numpy as np
import pandas as pd
import pytest

# Import strategy modules so that the @register_strategy decorators fire and
# populate the StrategyRegistry before any test runs.
import src.simulation.strategy  # noqa: F401
from src.simulation.strategy.adaptive import AdaptiveThresholdStrategy
from src.simulation.strategy.bb_reversion import BollingerReversionStrategy
from src.simulation.strategy.coinflip import Coinflip
from src.simulation.strategy.contrarian import ContrarianStrategy
from src.simulation.strategy.directional import DirectionalTradingStrategy
from src.simulation.strategy.ema_cross import EMACrossStrategy
from src.simulation.strategy.hold_days import HoldDaysStrategy
from src.simulation.strategy.hybrid_trend_ml import HybridTrendMLStrategy
from src.simulation.strategy.momentum import MomentumStrategy
from src.simulation.strategy.sltp import StopLossTakeProfitStrategy
from src.simulation.strategy.sma_trend import SMATrendStrategy
from src.simulation.strategy.strategy_registry import StrategyRegistry
from src.simulation.strategy.time_stop import TimeStopStrategy
from src.simulation.strategy.volatility_adjusted_threshold import VolatilityAdjustedStrategy


# ---------------------------------------------------------------------------
# Fake simulator stub
# ---------------------------------------------------------------------------


class _Sim:
    """Minimal simulator required by Strategy base class."""

    def __init__(self):
        self.transaction_cost = 0.001
        self.trades = []


# ---------------------------------------------------------------------------
# Helper: execute a strategy over a short price series and return final state
# ---------------------------------------------------------------------------


def _run_strategy(strategy, n=30, seed=0):
    """Drive strategy.execute() over n steps using deterministic data."""
    rng = np.random.default_rng(seed)
    prices = np.clip(100 + np.cumsum(rng.normal(0, 1, n)), 10, None)
    preds = rng.normal(0, 0.01, n)
    actuals = np.diff(prices, prepend=prices[0]) / prices
    dates = pd.date_range("2022-01-01", periods=n, freq="D")

    for i in range(n):
        strategy.execute(
            date=dates[i],
            price=float(prices[i]),
            pred_return=float(preds[i]),
            actual_return=float(actuals[i]),
        )
    return strategy


# ---------------------------------------------------------------------------
# StrategyRegistry
# ---------------------------------------------------------------------------


class TestStrategyRegistry:
    def test_all_expected_strategies_registered(self):
        registered = StrategyRegistry.list()
        expected = {
            "directional",
            "adaptive",
            "contrarian",
            "sltp",
            "hold_days",
            "time_stop",
            "coinflip",
            "momentum",
            "ema_cross",
            "sma_trend",
            "bb_reversion",
            "hybrid_trend_ml",
            "volatility_adjusted",
        }
        assert expected.issubset(set(registered))

    def test_get_returns_correct_class(self):
        cls = StrategyRegistry.get("directional")
        assert cls is DirectionalTradingStrategy

    def test_get_unknown_raises(self):
        with pytest.raises(ValueError, match="not registered"):
            StrategyRegistry.get("nonexistent_strategy_xyz")

    def test_register_duplicate_raises(self):
        with pytest.raises(ValueError, match="already registered"):
            StrategyRegistry.register("directional", DirectionalTradingStrategy)

    def test_create_returns_strategy_instance(self):
        sim = _Sim()
        strategy = StrategyRegistry.create("directional", simulator=sim, capital=10_000)
        assert isinstance(strategy, DirectionalTradingStrategy)

    def test_list_returns_list(self):
        assert isinstance(StrategyRegistry.list(), list)


# ---------------------------------------------------------------------------
# DirectionalTradingStrategy
# ---------------------------------------------------------------------------


class TestDirectionalTradingStrategy:
    def test_buys_on_positive_prediction(self):
        sim = _Sim()
        s = DirectionalTradingStrategy(sim, capital=10_000)
        s.execute(date=0, price=100.0, pred_return=0.01, actual_return=0.0)
        assert s.position == "long"

    def test_no_buy_on_negative_prediction(self):
        sim = _Sim()
        s = DirectionalTradingStrategy(sim, capital=10_000)
        s.execute(date=0, price=100.0, pred_return=-0.01, actual_return=0.0)
        assert s.position is None

    def test_sells_when_prediction_turns_negative(self):
        sim = _Sim()
        s = DirectionalTradingStrategy(sim, capital=10_000)
        s.execute(date=0, price=100.0, pred_return=0.01, actual_return=0.0)
        assert s.position == "long"
        s.execute(date=1, price=105.0, pred_return=-0.01, actual_return=0.0)
        assert s.position is None

    def test_execute_returns_four_tuple(self):
        sim = _Sim()
        s = DirectionalTradingStrategy(sim, capital=10_000)
        result = s.execute(date=0, price=100.0, pred_return=0.01, actual_return=0.0)
        assert len(result) == 4

    def test_get_minimum_data_points_default(self):
        assert DirectionalTradingStrategy.get_minimum_data_points() == 10


# ---------------------------------------------------------------------------
# AdaptiveThresholdStrategy
# ---------------------------------------------------------------------------


class TestAdaptiveThresholdStrategy:
    def test_buys_above_threshold(self):
        sim = _Sim()
        s = AdaptiveThresholdStrategy(sim, capital=10_000, threshold=0.005)
        s.execute(date=0, price=100.0, pred_return=0.01, actual_return=0.0)
        assert s.position == "long"

    def test_no_buy_below_threshold(self):
        sim = _Sim()
        s = AdaptiveThresholdStrategy(sim, capital=10_000, threshold=0.02)
        s.execute(date=0, price=100.0, pred_return=0.01, actual_return=0.0)
        assert s.position is None

    def test_sells_when_prediction_below_negative_threshold(self):
        sim = _Sim()
        s = AdaptiveThresholdStrategy(sim, capital=10_000, threshold=0.005)
        s.execute(date=0, price=100.0, pred_return=0.01, actual_return=0.0)
        s.execute(date=1, price=105.0, pred_return=-0.01, actual_return=0.0)
        assert s.position is None

    def test_default_threshold(self):
        sim = _Sim()
        s = AdaptiveThresholdStrategy(sim, capital=10_000)
        assert s.threshold == 0.02


# ---------------------------------------------------------------------------
# ContrarianStrategy
# ---------------------------------------------------------------------------


class TestContrarianStrategy:
    def test_buys_on_negative_prediction(self):
        sim = _Sim()
        s = ContrarianStrategy(sim, capital=10_000, threshold=-0.005)
        s.execute(date=0, price=100.0, pred_return=-0.01, actual_return=0.0)
        assert s.position == "long"

    def test_no_buy_on_positive_prediction(self):
        sim = _Sim()
        s = ContrarianStrategy(sim, capital=10_000, threshold=-0.02)
        s.execute(date=0, price=100.0, pred_return=0.01, actual_return=0.0)
        assert s.position is None

    def test_sells_after_contrarian_entry(self):
        sim = _Sim()
        s = ContrarianStrategy(sim, capital=10_000, threshold=-0.005)
        s.execute(date=0, price=100.0, pred_return=-0.01, actual_return=0.0)
        assert s.position == "long"
        # sell signal: pred_return > -threshold = 0.005
        s.execute(date=1, price=110.0, pred_return=0.01, actual_return=0.0)
        assert s.position is None


# ---------------------------------------------------------------------------
# StopLossTakeProfitStrategy
# ---------------------------------------------------------------------------


class TestStopLossTakeProfitStrategy:
    def test_buys_on_positive_prediction(self):
        sim = _Sim()
        s = StopLossTakeProfitStrategy(sim, capital=10_000)
        s.execute(date=0, price=100.0, pred_return=0.01, actual_return=0.0)
        assert s.position == "long"

    def test_sells_on_take_profit(self):
        sim = _Sim()
        s = StopLossTakeProfitStrategy(sim, capital=10_000, stop_loss=0.05, take_profit=0.04)
        s.execute(date=0, price=100.0, pred_return=0.01, actual_return=0.0)
        # price rose 5 % → pnl = 0.05 >= tp=0.04
        s.execute(date=1, price=105.0, pred_return=0.0, actual_return=0.0)
        assert s.position is None

    def test_sells_on_stop_loss(self):
        sim = _Sim()
        s = StopLossTakeProfitStrategy(sim, capital=10_000, stop_loss=0.03, take_profit=0.10)
        s.execute(date=0, price=100.0, pred_return=0.01, actual_return=0.0)
        # price fell 4 % → pnl = -0.04 <= -sl=-0.03
        s.execute(date=1, price=96.0, pred_return=0.0, actual_return=0.0)
        assert s.position is None

    def test_holds_within_band(self):
        sim = _Sim()
        s = StopLossTakeProfitStrategy(sim, capital=10_000, stop_loss=0.05, take_profit=0.10)
        s.execute(date=0, price=100.0, pred_return=0.01, actual_return=0.0)
        # price moved only 1 % → still within band
        s.execute(date=1, price=101.0, pred_return=0.0, actual_return=0.0)
        assert s.position == "long"


# ---------------------------------------------------------------------------
# HoldDaysStrategy
# ---------------------------------------------------------------------------


class TestHoldDaysStrategy:
    def test_buys_on_positive_prediction_above_threshold(self):
        sim = _Sim()
        s = HoldDaysStrategy(sim, capital=10_000, hold_days=5, threshold=0.005)
        s.execute(date=0, price=100.0, pred_return=0.01, actual_return=0.0)
        assert s.position == "long"

    def test_sells_after_hold_days(self):
        sim = _Sim()
        s = HoldDaysStrategy(sim, capital=10_000, hold_days=3, threshold=0.005)
        # Call 0: buy. Calls 1-3: each increments hold_counter; at counter==3 → sell.
        s.execute(date=0, price=100.0, pred_return=0.01, actual_return=0.0)
        for i in range(1, 4):
            s.execute(date=i, price=100.0, pred_return=0.0, actual_return=0.0)
        assert s.position is None

    def test_sells_early_on_negative_prediction(self):
        sim = _Sim()
        s = HoldDaysStrategy(sim, capital=10_000, hold_days=10, threshold=0.005)
        s.execute(date=0, price=100.0, pred_return=0.01, actual_return=0.0)
        s.execute(date=1, price=100.0, pred_return=-0.02, actual_return=0.0)
        assert s.position is None

    def test_get_minimum_data_points_default(self):
        assert HoldDaysStrategy.get_minimum_data_points() == 10


# ---------------------------------------------------------------------------
# TimeStopStrategy
# ---------------------------------------------------------------------------


class TestTimeStopStrategy:
    def test_buys_on_positive_prediction(self):
        sim = _Sim()
        s = TimeStopStrategy(sim, capital=10_000, hold_days=5)
        s.execute(date=0, price=100.0, pred_return=0.01, actual_return=0.0)
        assert s.position == "long"

    def test_sells_after_hold_days(self):
        sim = _Sim()
        s = TimeStopStrategy(sim, capital=10_000, hold_days=3)
        # Call 0: buy. Calls 1-3: each increments hold_counter; at counter==3 → sell.
        s.execute(date=0, price=100.0, pred_return=0.01, actual_return=0.0)
        for i in range(1, 4):
            s.execute(date=i, price=100.0, pred_return=0.0, actual_return=0.0)
        assert s.position is None

    def test_no_buy_on_negative_prediction(self):
        sim = _Sim()
        s = TimeStopStrategy(sim, capital=10_000, hold_days=5)
        s.execute(date=0, price=100.0, pred_return=-0.01, actual_return=0.0)
        assert s.position is None


# ---------------------------------------------------------------------------
# Coinflip
# ---------------------------------------------------------------------------


class TestCoinflip:
    def test_execute_returns_four_tuple(self):
        sim = _Sim()
        s = Coinflip(sim, capital=10_000)
        result = s.execute(date=0, price=100.0, pred_return=0.0, actual_return=0.0)
        assert len(result) == 4

    def test_runs_without_error_over_many_steps(self):
        random.seed(42)
        sim = _Sim()
        s = Coinflip(sim, capital=10_000)
        for i in range(50):
            s.execute(date=i, price=100.0, pred_return=0.0, actual_return=0.0)

    def test_capital_remains_positive(self):
        random.seed(0)
        sim = _Sim()
        s = Coinflip(sim, capital=10_000)
        for i in range(30):
            s.execute(date=i, price=100.0 + i, pred_return=0.0, actual_return=0.0)
        portfolio = s.shares * 129 if s.position == "long" else s.capital
        assert portfolio >= 0


# ---------------------------------------------------------------------------
# MomentumStrategy
# ---------------------------------------------------------------------------


class TestMomentumStrategy:
    def _make_roc_dict(self, n=30, seed=0):
        rng = np.random.default_rng(seed)
        dates = pd.date_range("2022-01-01", periods=n, freq="D")
        values = rng.normal(0, 0.01, n)
        return dict(zip(dates, values))

    def test_buys_when_roc_positive(self):
        sim = _Sim()
        dates = pd.date_range("2022-01-01", periods=1, freq="D")
        roc = {dates[0]: 0.02}
        s = MomentumStrategy(sim, capital=10_000, roc_series=roc)
        s.execute(date=dates[0], price=100.0, pred_return=0.0, actual_return=0.0)
        assert s.position == "long"

    def test_sells_when_roc_negative(self):
        sim = _Sim()
        dates = pd.date_range("2022-01-01", periods=2, freq="D")
        roc = {dates[0]: 0.02, dates[1]: -0.02}
        s = MomentumStrategy(sim, capital=10_000, roc_series=roc)
        s.execute(date=dates[0], price=100.0, pred_return=0.0, actual_return=0.0)
        s.execute(date=dates[1], price=105.0, pred_return=0.0, actual_return=0.0)
        assert s.position is None

    def test_skips_step_when_roc_missing(self):
        sim = _Sim()
        roc = {}  # no values
        s = MomentumStrategy(sim, capital=10_000, roc_series=roc)
        s.execute(date=pd.Timestamp("2022-01-01"), price=100.0, pred_return=0.0, actual_return=0.0)
        assert s.position is None

    def test_get_extra_params_returns_roc_series(self):
        prices = pd.Series(
            np.cumsum(np.abs(np.random.default_rng(0).normal(0, 1, 50))) + 100,
            index=pd.date_range("2022-01-01", periods=50, freq="D"),
        )
        params = MomentumStrategy.get_extra_params(prices)
        assert "roc_series" in params

    def test_get_minimum_data_points(self):
        assert MomentumStrategy.get_minimum_data_points() == 12


# ---------------------------------------------------------------------------
# EMACrossStrategy
# ---------------------------------------------------------------------------


class TestEMACrossStrategy:
    def _make_ema_dicts(self, n=30):
        dates = pd.date_range("2022-01-01", periods=n, freq="D")
        fast = {d: 105.0 for d in dates}
        slow = {d: 100.0 for d in dates}
        return fast, slow

    def test_buys_when_fast_above_slow(self):
        sim = _Sim()
        fast, slow = self._make_ema_dicts(1)
        date = pd.Timestamp("2022-01-01")
        s = EMACrossStrategy(sim, capital=10_000, ema_fast=fast, ema_slow=slow)
        s.execute(date=date, price=100.0, pred_return=0.0, actual_return=0.0)
        assert s.position == "long"

    def test_sells_when_fast_below_slow(self):
        sim = _Sim()
        dates = pd.date_range("2022-01-01", periods=2, freq="D")
        fast = {dates[0]: 105.0, dates[1]: 95.0}
        slow = {dates[0]: 100.0, dates[1]: 100.0}
        s = EMACrossStrategy(sim, capital=10_000, ema_fast=fast, ema_slow=slow)
        s.execute(date=dates[0], price=100.0, pred_return=0.0, actual_return=0.0)
        s.execute(date=dates[1], price=100.0, pred_return=0.0, actual_return=0.0)
        assert s.position is None

    def test_skips_when_ema_missing(self):
        sim = _Sim()
        s = EMACrossStrategy(sim, capital=10_000, ema_fast={}, ema_slow={})
        s.execute(date=pd.Timestamp("2022-01-01"), price=100.0, pred_return=0.0, actual_return=0.0)
        assert s.position is None

    def test_get_extra_params_has_ema_keys(self):
        prices = pd.Series(
            np.cumsum(np.abs(np.random.default_rng(0).normal(0, 1, 50))) + 100,
            index=pd.date_range("2022-01-01", periods=50, freq="D"),
        )
        params = EMACrossStrategy.get_extra_params(prices)
        assert "ema_fast" in params
        assert "ema_slow" in params

    def test_get_minimum_data_points(self):
        assert EMACrossStrategy.get_minimum_data_points() == 26


# ---------------------------------------------------------------------------
# SMATrendStrategy
# ---------------------------------------------------------------------------


class TestSMATrendStrategy:
    def test_buys_when_pred_positive_and_above_sma(self):
        sim = _Sim()
        date = pd.Timestamp("2022-01-01")
        sma = {date: 95.0}
        s = SMATrendStrategy(sim, capital=10_000, sma_series=sma)
        s.execute(date=date, price=100.0, pred_return=0.01, actual_return=0.0)
        assert s.position == "long"

    def test_no_buy_when_price_below_sma(self):
        sim = _Sim()
        date = pd.Timestamp("2022-01-01")
        sma = {date: 110.0}
        s = SMATrendStrategy(sim, capital=10_000, sma_series=sma)
        s.execute(date=date, price=100.0, pred_return=0.01, actual_return=0.0)
        assert s.position is None

    def test_sells_when_price_crosses_below_sma(self):
        sim = _Sim()
        dates = pd.date_range("2022-01-01", periods=2, freq="D")
        sma = {dates[0]: 95.0, dates[1]: 105.0}
        s = SMATrendStrategy(sim, capital=10_000, sma_series=sma)
        s.execute(date=dates[0], price=100.0, pred_return=0.01, actual_return=0.0)
        s.execute(date=dates[1], price=100.0, pred_return=0.0, actual_return=0.0)
        assert s.position is None

    def test_get_extra_params_has_sma_series(self):
        prices = pd.Series(
            np.cumsum(np.abs(np.random.default_rng(0).normal(0, 1, 60))) + 100,
            index=pd.date_range("2022-01-01", periods=60, freq="D"),
        )
        params = SMATrendStrategy.get_extra_params(prices)
        assert "sma_series" in params

    def test_get_minimum_data_points(self):
        assert SMATrendStrategy.get_minimum_data_points() == 50


# ---------------------------------------------------------------------------
# BollingerReversionStrategy
# ---------------------------------------------------------------------------


class TestBollingerReversionStrategy:
    def test_buys_when_price_below_lower_band(self):
        sim = _Sim()
        date = pd.Timestamp("2022-01-01")
        lower = {date: 105.0}
        upper = {date: 115.0}
        s = BollingerReversionStrategy(sim, capital=10_000, lower_band=lower, upper_band=upper)
        s.execute(date=date, price=100.0, pred_return=0.0, actual_return=0.0)
        assert s.position == "long"

    def test_sells_when_price_above_upper_band(self):
        sim = _Sim()
        dates = pd.date_range("2022-01-01", periods=2, freq="D")
        lower = {dates[0]: 105.0, dates[1]: 105.0}
        upper = {dates[0]: 115.0, dates[1]: 115.0}
        s = BollingerReversionStrategy(sim, capital=10_000, lower_band=lower, upper_band=upper)
        s.execute(date=dates[0], price=100.0, pred_return=0.0, actual_return=0.0)
        s.execute(date=dates[1], price=120.0, pred_return=0.0, actual_return=0.0)
        assert s.position is None

    def test_skips_when_bands_missing(self):
        sim = _Sim()
        s = BollingerReversionStrategy(sim, capital=10_000, lower_band={}, upper_band={})
        s.execute(date=pd.Timestamp("2022-01-01"), price=100.0, pred_return=0.0, actual_return=0.0)
        assert s.position is None

    def test_get_extra_params_has_band_keys(self):
        prices = pd.Series(
            np.cumsum(np.abs(np.random.default_rng(0).normal(0, 1, 30))) + 100,
            index=pd.date_range("2022-01-01", periods=30, freq="D"),
        )
        params = BollingerReversionStrategy.get_extra_params(prices)
        assert "upper_band" in params
        assert "lower_band" in params

    def test_get_minimum_data_points(self):
        assert BollingerReversionStrategy.get_minimum_data_points() == 20


# ---------------------------------------------------------------------------
# HybridTrendMLStrategy
# ---------------------------------------------------------------------------


class TestHybridTrendMLStrategy:
    def test_buys_when_pred_above_threshold_and_price_above_sma(self):
        sim = _Sim()
        date = pd.Timestamp("2022-01-01")
        sma = {date: 90.0}
        s = HybridTrendMLStrategy(sim, capital=10_000, sma_series=sma, threshold=0.005)
        s.execute(date=date, price=100.0, pred_return=0.01, actual_return=0.0)
        assert s.position == "long"

    def test_no_buy_when_price_below_sma(self):
        sim = _Sim()
        date = pd.Timestamp("2022-01-01")
        sma = {date: 110.0}
        s = HybridTrendMLStrategy(sim, capital=10_000, sma_series=sma, threshold=0.005)
        s.execute(date=date, price=100.0, pred_return=0.01, actual_return=0.0)
        assert s.position is None

    def test_sells_when_prediction_below_negative_threshold(self):
        sim = _Sim()
        dates = pd.date_range("2022-01-01", periods=2, freq="D")
        sma = {dates[0]: 90.0, dates[1]: 90.0}
        s = HybridTrendMLStrategy(sim, capital=10_000, sma_series=sma, threshold=0.005)
        s.execute(date=dates[0], price=100.0, pred_return=0.01, actual_return=0.0)
        s.execute(date=dates[1], price=100.0, pred_return=-0.02, actual_return=0.0)
        assert s.position is None

    def test_get_minimum_data_points(self):
        assert HybridTrendMLStrategy.get_minimum_data_points() == 50


# ---------------------------------------------------------------------------
# VolatilityAdjustedStrategy
# ---------------------------------------------------------------------------


class TestVolatilityAdjustedStrategy:
    def test_buys_when_pred_exceeds_vol_threshold(self):
        sim = _Sim()
        date = pd.Timestamp("2022-01-01")
        vol = {date: 0.005}  # threshold = k*vol = 0.5*0.005 = 0.0025
        s = VolatilityAdjustedStrategy(sim, capital=10_000, k=0.5, vol_series=vol)
        s.execute(date=date, price=100.0, pred_return=0.01, actual_return=0.0)
        assert s.position == "long"

    def test_sells_when_pred_below_negative_vol_threshold(self):
        sim = _Sim()
        dates = pd.date_range("2022-01-01", periods=2, freq="D")
        vol = {dates[0]: 0.005, dates[1]: 0.005}
        s = VolatilityAdjustedStrategy(sim, capital=10_000, k=0.5, vol_series=vol)
        s.execute(date=dates[0], price=100.0, pred_return=0.01, actual_return=0.0)
        s.execute(date=dates[1], price=100.0, pred_return=-0.01, actual_return=0.0)
        assert s.position is None

    def test_uses_default_vol_when_missing(self):
        sim = _Sim()
        s = VolatilityAdjustedStrategy(sim, capital=10_000, k=0.5, vol_series={})
        # pred_return=0.01 > k*default_vol=0.5*0.01=0.005 → should buy
        s.execute(date=pd.Timestamp("2022-01-01"), price=100.0, pred_return=0.01, actual_return=0.0)
        assert s.position == "long"

    def test_get_extra_params_has_vol_series(self):
        prices = pd.Series(
            np.cumsum(np.abs(np.random.default_rng(0).normal(0, 1, 30))) + 100,
            index=pd.date_range("2022-01-01", periods=30, freq="D"),
        )
        params = VolatilityAdjustedStrategy.get_extra_params(prices)
        assert "vol_series" in params

    def test_get_minimum_data_points(self):
        assert VolatilityAdjustedStrategy.get_minimum_data_points() == 20


# ---------------------------------------------------------------------------
# Strategy base class: buy / sell mechanics
# ---------------------------------------------------------------------------


class TestStrategyBuySellMechanics:
    _PRICE = 100.0
    _CAPITAL = 10_000.0

    def test_buy_sets_long_position(self):
        sim = _Sim()
        s = DirectionalTradingStrategy(sim, capital=self._CAPITAL)
        s.buy(date=0, price=self._PRICE)
        assert s.position == "long"
        assert s.shares > 0
        assert s.capital == 0

    def test_sell_clears_position(self):
        sim = _Sim()
        s = DirectionalTradingStrategy(sim, capital=self._CAPITAL)
        s.buy(date=0, price=self._PRICE)
        s.sell(date=1, price=110.0, pred_return=0.0)
        assert s.position is None
        assert s.shares == 0

    def test_sell_appends_to_trades(self):
        sim = _Sim()
        s = DirectionalTradingStrategy(sim, capital=self._CAPITAL)
        s.buy(date=0, price=self._PRICE)
        initial_trades = len(sim.trades)
        s.sell(date=1, price=110.0, pred_return=0.0)
        assert len(sim.trades) == initial_trades + 1

    def test_buy_appends_trade_record(self):
        sim = _Sim()
        s = DirectionalTradingStrategy(sim, capital=self._CAPITAL)
        s.buy(date=0, price=self._PRICE)
        assert len(sim.trades) == 1

    def test_shares_computed_correctly_on_buy(self):
        sim = _Sim()
        s = DirectionalTradingStrategy(sim, capital=self._CAPITAL)
        s.buy(date=0, price=self._PRICE)
        expected_shares = (self._CAPITAL * (1 - sim.transaction_cost)) / self._PRICE
        assert s.shares == pytest.approx(expected_shares, rel=1e-9)
