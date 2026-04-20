import numpy as np
import pandas as pd

EMA_FAST_PERIOD = 12
EMA_SLOW_PERIOD = 26


def ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Ratio of two series"""
    return numerator / denominator.replace(0, np.nan)


def std(series: pd.Series, window: int) -> pd.Series:
    """Standard deviation of a series"""
    return series.rolling(window).std()


def wma(series: pd.Series, window: int) -> pd.Series:
    """Weighted moving average"""
    weights = np.arange(1, window + 1, dtype=float)
    weights /= weights.sum()
    result = np.convolve(series.values, weights[::-1], mode="full")[: len(series)]
    result[: window - 1] = np.nan
    return pd.Series(result, index=series.index)


def hma(series: pd.Series, window: int) -> pd.Series:
    """Hull moving average"""
    half_length = int(window / 2)
    sqrt_length = int(np.sqrt(window))
    wma_half = wma(series, half_length)
    wma_full = wma(series, window)
    diff = 2 * wma_half - wma_full
    hma = wma(diff, sqrt_length)
    return hma


def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average"""
    return series.rolling(window).mean()


def ema(series: pd.Series, window: int) -> pd.Series:
    """Exponential moving average"""
    return series.ewm(span=window, adjust=False).mean()


def macd(
    fast_series: pd.Series | None = None,
    slow_series: pd.Series | None = None,
    series: pd.Series | None = None,
    fast_period: int = EMA_FAST_PERIOD,
    slow_period: int = EMA_SLOW_PERIOD,
) -> pd.Series:
    """
    Moving Average Convergence Divergence
    Compute MACD either from:
     - precomputed fast/slow EMA series, OR
     - raw price series with periods
    """
    if fast_series is not None and slow_series is not None:
        return fast_series - slow_series
    if series is not None:
        return ema(series, fast_period) - ema(series, slow_period)

    raise ValueError("Must provide either series or fast_series and slow_series")


def macd_signal(macd_series: pd.Series, signal_period: int = 9) -> pd.Series:
    """MACD Signal Line"""
    return ema(macd_series, signal_period)


def macd_hist(macd_series: pd.Series, signal_series: pd.Series) -> pd.Series:
    """MACD Histogram"""
    return macd_series - signal_series


def macd_momentum(macd_series: pd.Series) -> pd.Series:
    """MACD Momentum"""
    return macd_series.diff()


def macd_cross(macd_series: pd.Series, signal_series: pd.Series) -> pd.Series:
    """MACD Crossover"""
    return (macd_series > signal_series).astype(int)


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window).mean()
    rs = ratio(gain, loss)
    return 100 - (100 / (1 + rs))


def rsi_overbought(rsi_series: pd.Series) -> pd.Series:
    """RSI Overbought Indicator"""
    return (rsi_series > 70).astype(int)


def rsi_oversold(rsi_series: pd.Series) -> pd.Series:
    """RSI Oversold Indicator"""
    return (rsi_series < 30).astype(int)


def cci(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
    """Commodity Channel Index"""
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(window).mean()
    # Approximate mean absolute deviation using std * sqrt(2/π) ≈ 0.7979 (assumes normal distribution)
    mean_dev = tp.rolling(window).std() * 0.7979
    return (tp - sma_tp) / (0.015 * mean_dev)


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Williams' %R"""
    highest_high = high.rolling(window).max()
    lowest_low = low.rolling(window).min()
    return -100 * (highest_high - close) / (highest_high - lowest_low).replace(0, np.nan)


def tsi(close: pd.Series, long: int = 25, short: int = 13) -> pd.Series:
    """
    True Strength Index
    Formula:
        momentum = Close.diff()
        TSI = 100 * (EMA(EMA(momentum, long), short) /
                     EMA(EMA(|momentum|, long), short))
    """
    momentum = close.diff()
    abs_momentum = momentum.abs()

    # Double smoothed momentum
    ema1 = ema(momentum, long)
    ema2 = ema(ema1, short)
    abs_ema1 = ema(abs_momentum, long)
    abs_ema2 = ema(abs_ema1, short)

    return 100 * ratio(ema2, abs_ema2)


def stochastic_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Stochastic Oscillator"""
    low_windowed = low.rolling(window).min()
    high_windowed = high.rolling(window).max()

    return 100 * ratio(close - low_windowed, high_windowed - low_windowed)


def bollinger_bands_upper(close: pd.Series, window: int = 20) -> pd.Series:
    """Bollinger Bands Upper Band"""
    return sma(close, window) + 2 * std(close, window)


def bollinger_bands_lower(close: pd.Series, window: int = 20) -> pd.Series:
    """Bollinger Bands Lower Band"""
    return sma(close, window) - 2 * std(close, window)


def bollinger_bands_middle(close: pd.Series, window: int = 20) -> pd.Series:
    """Bollinger Bands Middle Band"""
    return sma(close, window)


def bollinger_bands_width(bb_upper: pd.Series, bb_lower: pd.Series) -> pd.Series:
    """Bollinger Bands Width"""
    return bb_upper - bb_lower


def bollinger_bands_width_ratio(bb_width: pd.Series, bb_middle: pd.Series) -> pd.Series:
    """Bollinger Bands Width Ratio"""
    return ratio(bb_width, bb_middle)


def bollinger_bands_percent_b(close: pd.Series, bb_lower: pd.Series, bb_upper: pd.Series) -> pd.Series:
    """Bollinger Bands Price Position (%B)"""
    return ratio(close - bb_lower, bb_upper - bb_lower)


def bollinger_bands(
    close: pd.Series, window: int = 20
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands"""
    mid = sma(close, window)
    rolling_std = std(close, window)
    bb_upper = mid + 2 * rolling_std
    bb_lower = mid - 2 * rolling_std
    bb_middle = mid

    bb_width = bollinger_bands_width(bb_upper, bb_lower)
    bb_width_ratio = bollinger_bands_width_ratio(bb_width, bb_middle)
    bb_b = bollinger_bands_percent_b(close, bb_lower, bb_upper)

    return bb_upper, bb_lower, bb_middle, bb_width, bb_width_ratio, bb_b


def price_change(series: pd.Series, window: int) -> pd.Series:
    """Price Change over x days"""
    return series.pct_change(window)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average True Range (ATR)"""
    prev_close = close.shift(1)

    tr_high_low = high - low
    tr_high_close = (high - prev_close).abs()
    tr_low_close = (low - prev_close).abs()
    tr = pd.concat([tr_high_low, tr_high_close, tr_low_close], axis=1).max(axis=1)

    return tr.rolling(window).mean()


def lagged_return(series: pd.Series, window: int) -> pd.Series:
    """Lagged Returns"""
    return price_change(series, 1).shift(window)


def momentum(series: pd.Series, window: int) -> pd.Series:
    """Momentum"""
    return series - series.shift(window)


def rate_of_change(series: pd.Series, window: int) -> pd.Series:
    """Rate of Change"""
    shifted = series.shift(window)
    return ratio(series - shifted, shifted).replace(0, np.nan) * 100


def plus_di(
    high: pd.Series, low: pd.Series, atr: pd.Series, window: int = 14, using_diff_values: bool = False
) -> pd.Series:
    """Plus Directional Indicator"""
    high_diff = high if using_diff_values else high.diff()
    low_diff = low if using_diff_values else -low.diff()

    plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0.0)
    return 100 * ratio(plus_dm.rolling(window).sum(), atr)


def minus_di(
    high: pd.Series, low: pd.Series, atr_: pd.Series, window: int = 14, using_diff_values: bool = False
) -> pd.Series:
    """Minus Directional Indicator"""
    high_diff = high if using_diff_values else high.diff()
    low_diff = low if using_diff_values else -low.diff()

    minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0.0)
    return 100 * ratio(minus_dm.rolling(window).sum(), atr_)


def adx(plus_di_: pd.Series, minus_di_: pd.Series) -> pd.Series:
    """Average Directional Movement Index"""
    return 100 * ratio((plus_di_ - minus_di_).abs(), (plus_di_ + minus_di_))


def directional_indicators(
    high: pd.Series, low: pd.Series, atr_series: pd.Series, window: int = 14
) -> tuple[pd.Series, pd.Series, pd.Series]:
    high_diff = high.diff()
    low_diff = -low.diff()
    plus_di_ = plus_di(high_diff, low_diff, atr_series, window=window, using_diff_values=True)
    minus_di_ = minus_di(high_diff, low_diff, atr_series, window=window, using_diff_values=True)
    adx_ = adx(plus_di_, minus_di_)

    return plus_di_, minus_di_, adx_


def sharpe(returns: pd.Series, window: int) -> pd.Series:
    """Sharpe Ratio"""
    return ratio(returns.rolling(window).mean(), returns.rolling(window).std())


def bull_engulfing(open_: pd.Series, close: pd.Series) -> pd.Series:
    """Bull Engulfing Pattern"""
    return ((close > open_.shift(1)) & (open_ < close.shift(1))).astype(int)


def doji(open_: pd.Series, close: pd.Series, high: pd.Series, low: pd.Series) -> pd.Series:
    """Doji"""
    return (abs(close - open_) / (high - low).replace(0, np.nan) < 0.1).astype(int)


def candlestick_patterns(
    open_: pd.Series, close: pd.Series, high: pd.Series, low: pd.Series
) -> tuple[pd.Series, pd.Series]:
    """Candlestick Patterns"""
    return bull_engulfing(open_, close), doji(open_, close, high, low)
