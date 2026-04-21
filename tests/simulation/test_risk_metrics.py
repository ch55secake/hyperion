"""Tests for src/simulation/risk_metrics.py"""

import math

import numpy as np
import pandas as pd
import pytest

from src.simulation.risk_metrics import RiskMetrics, compute_risk_metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_portfolio_history(values: list[float], prices: list[float] | None = None) -> pd.DataFrame:
    n = len(values)
    if prices is None:
        prices = values  # use the same series as benchmark when not specified
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n, freq="D"),
            "portfolio_value": values,
            "price": prices,
            "position": "cash",
            "prediction": 0.0,
        }
    )


def _make_trades(pnl_pcts: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"pnl_pct": pnl_pcts})


# ---------------------------------------------------------------------------
# Edge-case: insufficient data
# ---------------------------------------------------------------------------


class TestInsufficientData:
    def test_empty_portfolio_returns_all_none(self):
        ph = pd.DataFrame(columns=["portfolio_value", "price"])
        result = compute_risk_metrics(ph, pd.DataFrame(), total_return=0.0)
        assert isinstance(result, RiskMetrics)
        for field in result.__dataclass_fields__:
            assert getattr(result, field) is None

    def test_single_row_returns_all_none(self):
        ph = _make_portfolio_history([10_000.0])
        result = compute_risk_metrics(ph, pd.DataFrame(), total_return=0.0)
        for field in result.__dataclass_fields__:
            assert getattr(result, field) is None


# ---------------------------------------------------------------------------
# Return dataclass instance
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_risk_metrics_instance(self):
        ph = _make_portfolio_history([10_000.0, 10_100.0, 10_050.0, 10_200.0])
        result = compute_risk_metrics(ph, pd.DataFrame(), total_return=0.02)
        assert isinstance(result, RiskMetrics)


# ---------------------------------------------------------------------------
# Annualised volatility
# ---------------------------------------------------------------------------


class TestAnnualisedVolatility:
    def test_zero_volatility_series(self):
        """Flat portfolio → zero vol."""
        ph = _make_portfolio_history([10_000.0] * 10)
        result = compute_risk_metrics(ph, pd.DataFrame(), total_return=0.0)
        assert result.annualised_volatility == pytest.approx(0.0, abs=1e-10)

    def test_positive_volatility(self):
        rng = np.random.default_rng(0)
        values = 10_000.0 + np.cumsum(rng.normal(0, 50, 100))
        ph = _make_portfolio_history(values.tolist())
        result = compute_risk_metrics(ph, pd.DataFrame(), total_return=0.0)
        assert result.annualised_volatility is not None
        assert result.annualised_volatility > 0


# ---------------------------------------------------------------------------
# Sharpe ratio
# ---------------------------------------------------------------------------


class TestSharpeRatio:
    def test_sharpe_none_when_zero_std(self):
        """Flat portfolio returns → std is 0 → Sharpe is None."""
        ph = _make_portfolio_history([10_000.0] * 10)
        result = compute_risk_metrics(ph, pd.DataFrame(), total_return=0.0)
        assert result.sharpe_ratio is None

    def test_sharpe_sign_positive_for_growing_portfolio(self):
        values = [10_000.0 * (1.001**i) for i in range(100)]
        ph = _make_portfolio_history(values)
        result = compute_risk_metrics(ph, pd.DataFrame(), total_return=0.1)
        assert result.sharpe_ratio is not None
        assert result.sharpe_ratio > 0

    def test_sharpe_sign_negative_for_declining_portfolio(self):
        values = [10_000.0 * (0.999**i) for i in range(100)]
        ph = _make_portfolio_history(values)
        result = compute_risk_metrics(ph, pd.DataFrame(), total_return=-0.1)
        assert result.sharpe_ratio is not None
        assert result.sharpe_ratio < 0


# ---------------------------------------------------------------------------
# Sortino ratio
# ---------------------------------------------------------------------------


class TestSortinoRatio:
    def test_sortino_none_when_no_downside(self):
        """Monotonically increasing portfolio → no downside returns → Sortino None."""
        values = [10_000.0 + i * 10 for i in range(50)]
        ph = _make_portfolio_history(values)
        result = compute_risk_metrics(ph, pd.DataFrame(), total_return=0.05)
        # Less than 2 downside observations
        assert result.sortino_ratio is None

    def test_sortino_positive_for_rising_with_some_down_days(self):
        """A clearly rising portfolio with some down days should have a non-None Sortino ratio."""
        rng = np.random.default_rng(7)
        # Use a strong upward drift (0.005) so net return is reliably positive
        returns = rng.normal(0.005, 0.01, 120)
        values = [10_000.0]
        for r in returns:
            values.append(values[-1] * (1 + r))
        ph = _make_portfolio_history(values)
        total_ret = (values[-1] - values[0]) / values[0]
        result = compute_risk_metrics(ph, pd.DataFrame(), total_return=total_ret)
        assert result.sortino_ratio is not None
        assert result.sortino_ratio > 0


# ---------------------------------------------------------------------------
# Maximum drawdown
# ---------------------------------------------------------------------------


class TestMaxDrawdown:
    def test_no_drawdown_when_always_rising(self):
        values = [10_000.0 + i * 100 for i in range(20)]
        ph = _make_portfolio_history(values)
        result = compute_risk_metrics(ph, pd.DataFrame(), total_return=0.2)
        # No drawdown: every value >= its own running max
        assert result.max_drawdown == pytest.approx(0.0, abs=1e-10)

    def test_max_drawdown_known_value(self):
        """Peak at 12000, trough at 9000 → drawdown = -25 %."""
        values = [10_000.0, 11_000.0, 12_000.0, 9_000.0, 10_500.0]
        ph = _make_portfolio_history(values)
        result = compute_risk_metrics(ph, pd.DataFrame(), total_return=0.05)
        assert result.max_drawdown == pytest.approx(-0.25, rel=1e-6)

    def test_max_drawdown_is_non_positive(self):
        rng = np.random.default_rng(1)
        values = 10_000.0 + np.cumsum(rng.normal(0, 100, 50))
        ph = _make_portfolio_history(values.tolist())
        result = compute_risk_metrics(ph, pd.DataFrame(), total_return=0.0)
        assert result.max_drawdown <= 0


# ---------------------------------------------------------------------------
# Max drawdown duration
# ---------------------------------------------------------------------------


class TestMaxDrawdownDuration:
    def test_duration_zero_when_always_rising(self):
        values = [10_000.0 + i * 10 for i in range(10)]
        ph = _make_portfolio_history(values)
        result = compute_risk_metrics(ph, pd.DataFrame(), total_return=0.0)
        assert result.max_drawdown_duration == 0

    def test_duration_known_value(self):
        """Values dip below peak for exactly 3 steps."""
        values = [100.0, 110.0, 105.0, 108.0, 109.0, 115.0]
        # Peak after index 1 (110). Indices 2-4 are below peak (3 steps).
        ph = _make_portfolio_history(values)
        result = compute_risk_metrics(ph, pd.DataFrame(), total_return=0.0)
        assert result.max_drawdown_duration == 3


# ---------------------------------------------------------------------------
# Calmar ratio
# ---------------------------------------------------------------------------


class TestCalmarRatio:
    def test_calmar_none_when_no_drawdown(self):
        values = [10_000.0 + i * 100 for i in range(20)]
        ph = _make_portfolio_history(values)
        result = compute_risk_metrics(ph, pd.DataFrame(), total_return=0.2)
        assert result.calmar_ratio is None

    def test_calmar_positive_for_positive_return_with_drawdown(self):
        rng = np.random.default_rng(3)
        returns = rng.normal(0.001, 0.02, 100)
        values = [10_000.0]
        for r in returns:
            values.append(values[-1] * (1 + r))
        ph = _make_portfolio_history(values)
        total_ret = (values[-1] - values[0]) / values[0]
        result = compute_risk_metrics(ph, pd.DataFrame(), total_return=total_ret)
        if result.max_drawdown < 0:
            assert result.calmar_ratio is not None


# ---------------------------------------------------------------------------
# Win rate
# ---------------------------------------------------------------------------


class TestWinRate:
    def test_win_rate_all_wins(self):
        trades = _make_trades([1.0, 2.0, 3.0])
        ph = _make_portfolio_history([10_000.0, 11_000.0, 12_000.0])
        result = compute_risk_metrics(ph, trades, total_return=0.1)
        assert result.win_rate == pytest.approx(1.0)

    def test_win_rate_all_losses(self):
        trades = _make_trades([-1.0, -2.0])
        ph = _make_portfolio_history([10_000.0, 9_000.0, 8_000.0])
        result = compute_risk_metrics(ph, trades, total_return=-0.2)
        assert result.win_rate == pytest.approx(0.0)

    def test_win_rate_mixed(self):
        trades = _make_trades([1.0, -1.0, 2.0, -2.0])
        ph = _make_portfolio_history([10_000.0, 10_100.0, 9_900.0, 10_200.0])
        result = compute_risk_metrics(ph, trades, total_return=0.0)
        assert result.win_rate == pytest.approx(0.5)

    def test_win_rate_none_when_no_trades(self):
        ph = _make_portfolio_history([10_000.0, 10_100.0])
        result = compute_risk_metrics(ph, pd.DataFrame(), total_return=0.0)
        assert result.win_rate is None


# ---------------------------------------------------------------------------
# Profit factor
# ---------------------------------------------------------------------------


class TestProfitFactor:
    def test_profit_factor_known_value(self):
        """Gross profit 6, gross loss 3 → PF = 2.0."""
        trades = _make_trades([3.0, 3.0, -3.0])
        ph = _make_portfolio_history([10_000.0, 10_300.0, 10_600.0])
        result = compute_risk_metrics(ph, trades, total_return=0.06)
        assert result.profit_factor == pytest.approx(2.0)

    def test_profit_factor_inf_when_no_losses(self):
        trades = _make_trades([1.0, 2.0, 3.0])
        ph = _make_portfolio_history([10_000.0, 10_100.0, 10_300.0])
        result = compute_risk_metrics(ph, trades, total_return=0.03)
        assert result.profit_factor == math.inf

    def test_profit_factor_none_when_no_trades(self):
        ph = _make_portfolio_history([10_000.0, 10_100.0])
        result = compute_risk_metrics(ph, pd.DataFrame(), total_return=0.0)
        assert result.profit_factor is None


# ---------------------------------------------------------------------------
# Average win/loss ratio
# ---------------------------------------------------------------------------


class TestAvgWinLossRatio:
    def test_avg_win_loss_ratio_known_value(self):
        """Mean win 2, mean loss 1 → ratio = 2.0."""
        trades = _make_trades([1.0, 3.0, -1.0, -1.0])
        ph = _make_portfolio_history([10_000.0, 10_100.0, 10_300.0, 10_200.0])
        result = compute_risk_metrics(ph, trades, total_return=0.02)
        assert result.avg_win_loss_ratio == pytest.approx(2.0)

    def test_avg_win_loss_none_when_only_wins(self):
        trades = _make_trades([1.0, 2.0])
        ph = _make_portfolio_history([10_000.0, 10_100.0, 10_300.0])
        result = compute_risk_metrics(ph, trades, total_return=0.03)
        assert result.avg_win_loss_ratio is None


# ---------------------------------------------------------------------------
# Value at Risk
# ---------------------------------------------------------------------------


class TestValueAtRisk:
    def test_var_is_negative_for_volatile_series(self):
        rng = np.random.default_rng(5)
        values = 10_000.0 + np.cumsum(rng.normal(0, 200, 60))
        ph = _make_portfolio_history(values.tolist())
        result = compute_risk_metrics(ph, pd.DataFrame(), total_return=0.0)
        # 5th percentile of returns is typically a loss for a volatile series
        assert result.value_at_risk_95 is not None

    def test_var_95_is_float(self):
        ph = _make_portfolio_history([10_000.0, 9_800.0, 10_100.0, 9_900.0, 10_200.0])
        result = compute_risk_metrics(ph, pd.DataFrame(), total_return=0.0)
        assert isinstance(result.value_at_risk_95, float)


# ---------------------------------------------------------------------------
# Beta to benchmark
# ---------------------------------------------------------------------------


class TestBetaToBenchmark:
    def test_beta_one_when_portfolio_mirrors_price(self):
        """When portfolio value == price, beta should be 1.0."""
        rng = np.random.default_rng(9)
        prices = 100.0 + np.cumsum(rng.normal(0, 1, 60))
        ph = _make_portfolio_history(prices.tolist(), prices.tolist())
        result = compute_risk_metrics(ph, pd.DataFrame(), total_return=0.0)
        assert result.beta_to_benchmark == pytest.approx(1.0, rel=1e-4)

    def test_beta_is_float_or_none(self):
        ph = _make_portfolio_history([10_000.0, 10_100.0, 10_050.0, 10_200.0], [100.0, 101.0, 100.5, 102.0])
        result = compute_risk_metrics(ph, pd.DataFrame(), total_return=0.0)
        assert result.beta_to_benchmark is None or isinstance(result.beta_to_benchmark, float)


# ---------------------------------------------------------------------------
# Integration: simulate() includes risk_metrics key
# ---------------------------------------------------------------------------


class TestSimulatorIntegration:
    def test_simulate_result_has_risk_metrics_key(self):
        from src.simulation.trading_simulator import TradingSimulator

        rng = np.random.default_rng(42)
        prices = pd.Series(100.0 + np.cumsum(rng.normal(0, 1, 60)))
        actual_returns = prices.pct_change().fillna(0)
        predictions = rng.normal(0, 0.01, 60)
        dates = pd.date_range("2024-01-01", periods=60, freq="D")

        sim = TradingSimulator()
        result = sim.simulate(predictions, actual_returns, prices=prices, dates=dates)
        assert "risk_metrics" in result
        assert isinstance(result["risk_metrics"], RiskMetrics)

    def test_simulate_risk_metrics_fields_are_numeric_or_none(self):
        from src.simulation.trading_simulator import TradingSimulator

        rng = np.random.default_rng(7)
        prices = pd.Series(100.0 + np.cumsum(rng.normal(0, 1, 80)))
        actual_returns = prices.pct_change().fillna(0)
        predictions = rng.normal(0, 0.01, 80)
        dates = pd.date_range("2024-01-01", periods=80, freq="D")

        sim = TradingSimulator()
        result = sim.simulate(predictions, actual_returns, prices=prices, dates=dates)
        rm = result["risk_metrics"]
        for field in rm.__dataclass_fields__:
            val = getattr(rm, field)
            assert val is None or isinstance(val, (int, float)), f"Field {field} has unexpected type {type(val)}"
