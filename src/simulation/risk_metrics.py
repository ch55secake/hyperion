from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RiskMetrics:
    """Comprehensive risk metrics computed from a trading simulation."""

    sharpe_ratio: float | None
    sortino_ratio: float | None
    max_drawdown: float | None
    max_drawdown_duration: int | None
    calmar_ratio: float | None
    win_rate: float | None
    profit_factor: float | None
    avg_win_loss_ratio: float | None
    annualised_volatility: float | None
    value_at_risk_95: float | None
    beta_to_benchmark: float | None


def compute_risk_metrics(
    portfolio_history: pd.DataFrame,
    trades: pd.DataFrame,
    total_return: float,
    trading_days_per_year: int = 252,
) -> RiskMetrics:
    """
    Compute comprehensive risk metrics from simulation results.

    Parameters
    ----------
    portfolio_history:
        DataFrame with at least ``portfolio_value`` and ``price`` columns
        (one row per simulation step).
    trades:
        DataFrame with at least a ``pnl_pct`` column (one row per trade).
    total_return:
        Total strategy return as a decimal (e.g. 0.12 for 12 %).
    trading_days_per_year:
        Number of trading days used to annualise period metrics.  Defaults
        to 252 (US equity standard).

    Returns
    -------
    RiskMetrics
        Dataclass containing all computed metrics.  Individual fields are
        ``None`` when there is insufficient data to compute them.
    """
    _none = RiskMetrics(
        sharpe_ratio=None,
        sortino_ratio=None,
        max_drawdown=None,
        max_drawdown_duration=None,
        calmar_ratio=None,
        win_rate=None,
        profit_factor=None,
        avg_win_loss_ratio=None,
        annualised_volatility=None,
        value_at_risk_95=None,
        beta_to_benchmark=None,
    )

    if portfolio_history.empty or len(portfolio_history) < 2:
        return _none

    values = portfolio_history["portfolio_value"].to_numpy(dtype=float)
    prices = portfolio_history["price"].to_numpy(dtype=float)

    # Daily portfolio returns
    port_returns = np.diff(values) / np.where(values[:-1] != 0, values[:-1], np.nan)

    # --- Annualised volatility ---
    annualised_vol: float | None = None
    if len(port_returns) > 1:
        annualised_vol = float(np.nanstd(port_returns, ddof=1) * np.sqrt(trading_days_per_year))

    # --- Sharpe ratio (risk-free rate assumed zero) ---
    sharpe: float | None = None
    mean_ret = float(np.nanmean(port_returns))
    std_ret = float(np.nanstd(port_returns, ddof=1)) if len(port_returns) > 1 else 0.0
    if std_ret > 0:
        sharpe = mean_ret / std_ret * np.sqrt(trading_days_per_year)

    # --- Sortino ratio (downside deviation only) ---
    sortino: float | None = None
    downside = port_returns[port_returns < 0]
    if len(downside) > 1:
        downside_std = float(np.std(downside, ddof=1))
        if downside_std > 0:
            sortino = mean_ret / downside_std * np.sqrt(trading_days_per_year)

    # --- Maximum drawdown ---
    running_max = np.maximum.accumulate(values)
    drawdowns = np.where(running_max != 0, (values - running_max) / running_max, 0.0)
    max_dd = float(np.min(drawdowns))

    # --- Max drawdown duration (trading days spent below a previous peak) ---
    in_drawdown = values < running_max
    max_dd_duration = 0
    current_streak = 0
    for flag in in_drawdown:
        if flag:
            current_streak += 1
            if current_streak > max_dd_duration:
                max_dd_duration = current_streak
        else:
            current_streak = 0

    # --- Calmar ratio ---
    calmar: float | None = None
    n_days = len(values)
    if n_days > 1:
        annualised_return = (1 + total_return) ** (trading_days_per_year / n_days) - 1
        if max_dd < 0:
            calmar = annualised_return / abs(max_dd)

    # --- Trade-based metrics ---
    win_rate: float | None = None
    profit_factor: float | None = None
    avg_win_loss: float | None = None

    if not trades.empty and "pnl_pct" in trades.columns:
        pnls = trades["pnl_pct"].dropna()
        if len(pnls) > 0:
            wins = pnls[pnls > 0]
            losses = pnls[pnls < 0]
            win_rate = float(len(wins) / len(pnls))

            gross_profit = float(wins.sum())
            gross_loss = float(abs(losses.sum()))
            if gross_loss > 0:
                profit_factor = gross_profit / gross_loss
            elif gross_profit > 0:
                profit_factor = float("inf")

            if len(wins) > 0 and len(losses) > 0:
                avg_win_loss = float(wins.mean() / abs(losses.mean()))

    # --- Value at Risk (95 % — 5th percentile of daily returns) ---
    var_95: float | None = None
    if len(port_returns) > 0:
        var_95 = float(np.nanpercentile(port_returns, 5))

    # --- Beta to benchmark ---
    beta: float | None = None
    bh_returns = np.diff(prices) / np.where(prices[:-1] != 0, prices[:-1], np.nan)
    if len(port_returns) > 1 and len(bh_returns) == len(port_returns):
        bh_std = float(np.nanstd(bh_returns, ddof=1))
        if bh_std > 0:
            valid = ~(np.isnan(port_returns) | np.isnan(bh_returns))
            if valid.sum() > 1:
                cov_matrix = np.cov(port_returns[valid], bh_returns[valid])
                beta = float(cov_matrix[0, 1] / (bh_std**2))

    return RiskMetrics(
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_dd,
        max_drawdown_duration=max_dd_duration,
        calmar_ratio=calmar,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_win_loss_ratio=avg_win_loss,
        annualised_volatility=annualised_vol,
        value_at_risk_95=var_95,
        beta_to_benchmark=beta,
    )
