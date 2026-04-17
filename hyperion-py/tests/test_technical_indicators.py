"""Unit tests for src/feature/technical_indicators.py"""

import pandas as pd
import pytest

import src.feature.technical_indicators as ti


@pytest.fixture
def close_series():
    """A simple deterministic close price series."""
    prices = [
        10.0,
        11.0,
        10.5,
        12.0,
        11.5,
        13.0,
        12.5,
        14.0,
        13.5,
        15.0,
        14.5,
        16.0,
        15.5,
        17.0,
        16.5,
        18.0,
        17.5,
        19.0,
        18.5,
        20.0,
    ]
    return pd.Series(prices, dtype=float)


@pytest.fixture
def ohlcv():
    """A simple deterministic OHLCV series."""
    n = 30
    close = pd.Series([10 + i * 0.5 for i in range(n)], dtype=float)
    high = close + 0.5
    low = close - 0.5
    open_ = close - 0.1
    volume = pd.Series([1000.0 + i * 10 for i in range(n)], dtype=float)
    return open_, high, low, close, volume


# ---------------------------------------------------------------------------
# Moving averages
# ---------------------------------------------------------------------------


class TestSMA:
    def test_basic_values(self, close_series):
        result = ti.sma(close_series, 3)
        expected = close_series.rolling(3).mean()
        pd.testing.assert_series_equal(result, expected)

    def test_first_values_are_nan(self, close_series):
        result = ti.sma(close_series, 5)
        assert result.iloc[:4].isna().all()

    def test_window_1(self, close_series):
        result = ti.sma(close_series, 1)
        pd.testing.assert_series_equal(result, close_series.rolling(1).mean())


class TestEMA:
    def test_returns_series(self, close_series):
        result = ti.ema(close_series, 5)
        assert isinstance(result, pd.Series)
        assert len(result) == len(close_series)

    def test_no_nan_after_first(self, close_series):
        result = ti.ema(close_series, 3)
        # EMA with adjust=False should not produce NaN after index 0
        assert not result.iloc[1:].isna().any()


class TestWMA:
    def test_shape(self, close_series):
        result = ti.wma(close_series, 3)
        assert len(result) == len(close_series)

    def test_manual_calculation(self):
        s = pd.Series([1.0, 2.0, 3.0])
        result = ti.wma(s, 3)
        expected_last = (1 * 1 + 2 * 2 + 3 * 3) / (1 + 2 + 3)
        assert abs(result.iloc[-1] - expected_last) < 1e-9


class TestHMA:
    def test_returns_series(self, close_series):
        result = ti.hma(close_series, 6)
        assert isinstance(result, pd.Series)
        assert len(result) == len(close_series)


# ---------------------------------------------------------------------------
# Oscillators
# ---------------------------------------------------------------------------


class TestRSI:
    def test_range(self, close_series):
        result = ti.rsi(close_series, 5)
        valid = result.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_overbought_flag(self):
        # Constantly rising series should push RSI towards 100
        s = pd.Series([float(i) for i in range(1, 30)])
        rsi_val = ti.rsi(s, 14)
        overbought = ti.rsi_overbought(rsi_val)
        assert pd.api.types.is_integer_dtype(overbought)

    def test_oversold_flag(self):
        s = pd.Series([float(30 - i) for i in range(30)])
        rsi_val = ti.rsi(s, 14)
        oversold = ti.rsi_oversold(rsi_val)
        assert pd.api.types.is_integer_dtype(oversold)


class TestMACD:
    def test_from_precomputed(self, close_series):
        fast = ti.ema(close_series, 12)
        slow = ti.ema(close_series, 26)
        result = ti.macd(fast_series=fast, slow_series=slow)
        pd.testing.assert_series_equal(result, fast - slow)

    def test_from_series(self, close_series):
        result = ti.macd(series=close_series)
        assert isinstance(result, pd.Series)

    def test_raises_without_args(self):
        with pytest.raises(ValueError):
            ti.macd()

    def test_signal_line(self, close_series):
        fast = ti.ema(close_series, 12)
        slow = ti.ema(close_series, 26)
        macd_series = ti.macd(fast_series=fast, slow_series=slow)
        signal = ti.macd_signal(macd_series)
        assert isinstance(signal, pd.Series)
        assert len(signal) == len(macd_series)

    def test_histogram(self, close_series):
        fast = ti.ema(close_series, 12)
        slow = ti.ema(close_series, 26)
        macd_series = ti.macd(fast_series=fast, slow_series=slow)
        signal = ti.macd_signal(macd_series)
        hist = ti.macd_hist(macd_series, signal)
        expected = macd_series - signal
        pd.testing.assert_series_equal(hist, expected)

    def test_cross_binary(self, close_series):
        fast = ti.ema(close_series, 12)
        slow = ti.ema(close_series, 26)
        macd_s = ti.macd(fast_series=fast, slow_series=slow)
        signal = ti.macd_signal(macd_s)
        cross = ti.macd_cross(macd_s, signal)
        assert set(cross.dropna().unique()).issubset({0, 1})


class TestCCI:
    def test_returns_series(self, ohlcv):
        _, high, low, close, _ = ohlcv
        result = ti.cci(high, low, close)
        assert isinstance(result, pd.Series)
        assert len(result) == len(close)


class TestWilliamsR:
    def test_range(self, ohlcv):
        _, high, low, close, _ = ohlcv
        result = ti.williams_r(high, low, close)
        valid = result.dropna()
        assert (valid >= -100).all() and (valid <= 0).all()


class TestStochasticOscillator:
    def test_range(self, ohlcv):
        _, high, low, close, _ = ohlcv
        result = ti.stochastic_oscillator(high, low, close)
        valid = result.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()


class TestTSI:
    def test_returns_series(self, close_series):
        result = ti.tsi(close_series)
        assert isinstance(result, pd.Series)
        assert len(result) == len(close_series)


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------


class TestBollingerBands:
    def test_upper_greater_than_lower(self, close_series):
        upper, lower, _, _, _, _ = ti.bollinger_bands(close_series)
        valid = ~(upper.isna() | lower.isna())
        assert (upper[valid] >= lower[valid]).all()

    def test_middle_is_sma20(self, close_series):
        _, _, middle, _, _, _ = ti.bollinger_bands(close_series)
        expected = ti.sma(close_series, 20)
        pd.testing.assert_series_equal(middle, expected)

    def test_individual_helpers(self, close_series):
        upper = ti.bollinger_bands_upper(close_series)
        lower = ti.bollinger_bands_lower(close_series)
        middle = ti.bollinger_bands_middle(close_series)
        width = ti.bollinger_bands_width(upper, lower)
        expected_width = upper - lower
        pd.testing.assert_series_equal(width, expected_width)
        width_ratio = ti.bollinger_bands_width_ratio(width, middle)
        assert isinstance(width_ratio, pd.Series)
        pct_b = ti.bollinger_bands_percent_b(close_series, lower, upper)
        assert isinstance(pct_b, pd.Series)


# ---------------------------------------------------------------------------
# Price / Volume helpers
# ---------------------------------------------------------------------------


class TestPriceChange:
    def test_single_period(self):
        s = pd.Series([100.0, 110.0, 121.0])
        result = ti.price_change(s, 1)
        assert abs(result.iloc[-1] - 0.1) < 1e-9

    def test_multi_period(self):
        s = pd.Series([100.0, 110.0, 121.0])
        result = ti.price_change(s, 2)
        assert abs(result.iloc[-1] - 0.21) < 1e-9


class TestATR:
    def test_non_negative(self, ohlcv):
        _, high, low, close, _ = ohlcv
        result = ti.atr(high, low, close)
        valid = result.dropna()
        assert (valid >= 0).all()


class TestMomentum:
    def test_basic(self):
        s = pd.Series([10.0, 12.0, 15.0, 18.0])
        result = ti.momentum(s, 2)
        assert abs(result.iloc[-1] - 6.0) < 1e-9


class TestRateOfChange:
    def test_basic(self):
        s = pd.Series([100.0, 110.0, 121.0])
        result = ti.rate_of_change(s, 1)
        assert abs(result.iloc[-1] - 10.0) < 1e-9


class TestLaggedReturn:
    def test_lag_1(self):
        s = pd.Series([100.0, 110.0, 121.0])
        result = ti.lagged_return(s, 1)
        # lag-1 of pct_change(1) should shift once more
        assert isinstance(result, pd.Series)
        assert len(result) == len(s)


# ---------------------------------------------------------------------------
# Directional indicators
# ---------------------------------------------------------------------------


class TestDirectionalIndicators:
    def test_returns_three_series(self, ohlcv):
        _, high, low, close, _ = ohlcv
        atr_ = ti.atr(high, low, close)
        plus, minus, adx_ = ti.directional_indicators(high, low, atr_)
        assert isinstance(plus, pd.Series)
        assert isinstance(minus, pd.Series)
        assert isinstance(adx_, pd.Series)


# ---------------------------------------------------------------------------
# Sharpe
# ---------------------------------------------------------------------------


class TestSharpe:
    def test_returns_series(self, close_series):
        returns = ti.price_change(close_series, 1)
        result = ti.sharpe(returns, 5)
        assert isinstance(result, pd.Series)


# ---------------------------------------------------------------------------
# Candlestick patterns
# ---------------------------------------------------------------------------


class TestCandlestickPatterns:
    def test_binary_output(self, ohlcv):
        open_, high, low, close, _ = ohlcv
        bull, doji = ti.candlestick_patterns(open_, close, high, low)
        assert set(bull.unique()).issubset({0, 1})
        assert set(doji.unique()).issubset({0, 1})

    def test_doji_flat_bar(self):
        # When open == close and range is large, doji should be 1
        open_ = pd.Series([10.0])
        close = pd.Series([10.0])
        high = pd.Series([20.0])
        low = pd.Series([0.0])
        result = ti.doji(open_, close, high, low)
        assert result.iloc[0] == 1

    def test_bull_engulfing_pattern(self):
        # prev: close < open (bearish), curr: open < prev_close and close > prev_open
        open_ = pd.Series([12.0, 10.0])
        close = pd.Series([11.0, 13.0])
        result = ti.bull_engulfing(open_, close)
        assert result.iloc[1] == 1


# ---------------------------------------------------------------------------
# Ratio helper
# ---------------------------------------------------------------------------


class TestRatio:
    def test_basic(self):
        num = pd.Series([10.0, 20.0])
        den = pd.Series([2.0, 4.0])
        result = ti.ratio(num, den)
        pd.testing.assert_series_equal(result, pd.Series([5.0, 5.0]))

    def test_zero_denominator_produces_nan(self):
        num = pd.Series([10.0])
        den = pd.Series([0.0])
        result = ti.ratio(num, den)
        assert result.isna().iloc[0]
