from __future__ import annotations

import numpy as np
import pandas as pd

from pandas import DataFrame
from sklearn.preprocessing import StandardScaler

import src.feature.technical_indicators as ti
from src.feature.regime import classify_regime

_required_cols = ["Open", "High", "Low", "Close", "Volume"]
_all_cols = ["Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits"]
_all_cols_with_targets = ["Open", "High", "Low", "Close", "Volume", "Target", "Dividends", "Stock Splits"]
_ma_windows = [5, 10, 12, 20, 26, 50, 100]
_price_change_windows = [1, 5, 10, 20, 50, 100]
_volume_change_windows = [5, 10, 20, 50]
_volatility_windows = [5, 10, 20, 50]
_lagged_returns_windows = [1, 2, 3, 5, 10, 20]
_momentum_windows = [1, 5, 10, 20, 50]
_rate_of_change_windows = [5, 10, 20, 50]
_sharpe_windows = [5, 10, 20, 50, 100]


class FeatureEngineering:
    """Creates technical indicators and features for machine learning, robust to small/incomplete data."""

    def __init__(self, df: DataFrame):
        self.df: DataFrame = self._ensure_required_columns(df)
        self.__n_rows: int = len(self.df)
        self.__calculated: bool = False
        self.__df_prepared = None

    @staticmethod
    def _ensure_required_columns(df: DataFrame) -> DataFrame:
        for col in _required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        return df.sort_index().copy()

    def get_df(self) -> DataFrame:
        return self.df

    def _add_moving_averages(self) -> FeatureEngineering:
        for w in _ma_windows:
            if self.__n_rows >= w:
                # SMA_n: Simple Moving Average over n days. Measures the mean closing price over a period,
                # smoothing short-term fluctuations.
                self.df[f"SMA_{w}"] = ti.sma(self.df["Close"], w)
                # EMA_n: Exponential Moving Average over n days. Gives more weight to recent prices,
                # highlighting short-term trends.
                self.df[f"EMA_{w}"] = ti.ema(self.df["Close"], w)

                # WMA_n: Weighted Moving Average over n days. Similar to EMA but weights increase linearly;
                # less sensitive to extreme spikes.
                self.df[f"WMA_{w}"] = ti.wma(self.df["Close"], w)
                # HMA_n: Hull Moving Average over n days. Faster and less noisy than WMA;
                # responds more quickly to trend changes.
                self.df[f"HMA_{w}"] = ti.hma(self.df["Close"], w)

        return self

    def _add_safe_price_ratios(self) -> FeatureEngineering:
        # Price_to_SMA20 / Price_to_SMA50: Ratio of current price to 20/50-day SMA.
        # Shows if the stock is above/below short/medium-term trend.
        self.df["Price_to_SMA20"] = ti.ratio(self.df["Close"], self.df.get("SMA_20", pd.Series(1)))
        self.df["Price_to_SMA50"] = ti.ratio(self.df["Close"], self.df.get("SMA_50", pd.Series(1)))
        # Price_to_EMA12 / Price_to_EMA26: Ratio of current price to 12/26-day EMA.
        # Highlights recent momentum relative to longer trend.
        self.df["Price_to_EMA12"] = ti.ratio(self.df["Close"], self.df.get("EMA_12", pd.Series(1)))
        self.df["Price_to_EMA26"] = ti.ratio(self.df["Close"], self.df.get("EMA_26", pd.Series(1)))
        # EMA_12_26_Ratio: Ratio of EMA12 to EMA26. Indicator for trend strength (used in MACD calculation).
        if "EMA_12" in self.df.columns and "EMA_26" in self.df.columns:
            self.df["EMA_12_26_Ratio"] = ti.ratio(self.df["EMA_12"], self.df["EMA_26"])
        # SMA_5_20_Ratio / SMA_10_50_Ratio: Ratio of short-term SMA to medium-term SMA.
        # Signals short-term trend direction relative to longer trend.
        if "SMA_5" in self.df.columns and "SMA_20" in self.df.columns:
            self.df["SMA_5_20_Ratio"] = ti.ratio(self.df["SMA_5"], self.df["SMA_20"])
        if "SMA_10" in self.df.columns and "SMA_50" in self.df.columns:
            self.df["SMA_10_50_Ratio"] = ti.ratio(self.df["SMA_10"], self.df["SMA_50"])

        return self

    def _add_macd(self) -> FeatureEngineering:
        if "EMA_12" in self.df.columns and "EMA_26" in self.df.columns:
            # MACD: Difference between EMA12 and EMA26. Indicates trend strength and direction.
            self.df["MACD"] = ti.macd(self.df["EMA_12"], self.df["EMA_26"])
            # MACD_Signal: 9-day EMA of MACD. Signal line for MACD crossovers.
            self.df["MACD_Signal"] = ti.macd_signal(self.df["MACD"])
            # MACD_Hist: MACD minus signal line. Shows momentum acceleration/deceleration.
            self.df["MACD_Hist"] = ti.macd_hist(self.df["MACD"], self.df["MACD_Signal"])
            # MACD_Momentum: Day-to-day change in MACD. Measures momentum shift.
            self.df["MACD_Momentum"] = self.df["MACD"].diff()
            # MACD_Cross: Binary flag (1 if MACD > Signal). Indicates bullish (1) or bearish (0) signal.
            self.df["MACD_Cross"] = ti.macd_cross(self.df["MACD"], self.df["MACD_Signal"])

        return self

    def _add_rsi(self) -> FeatureEngineering:
        if self.__n_rows < 14:
            return self

        self.df["RSI"] = ti.rsi(self.df["Close"])
        # RSI_Overbought / RSI_Oversold: Binary indicators for RSI extremes.
        self.df["RSI_Overbought"] = ti.rsi_overbought(self.df["RSI"])
        self.df["RSI_Oversold"] = ti.rsi_oversold(self.df["RSI"])

        return self

    def _add_cci(self) -> FeatureEngineering:
        # CCI: Measures the deviation of the typical price ((High + Low + Close)/3) from its moving average
        # over a given window (default 20 days).
        self.df["CCI"] = ti.cci(self.df["High"], self.df["Low"], self.df["Close"])
        return self

    def _add_williams_r(self) -> FeatureEngineering:
        # WilliamsR: Momentum indicator that measures the current closing price relative to the highest high
        # and the lowest low over a given period (default 14 days).
        self.df["WilliamsR"] = ti.williams_r(self.df["High"], self.df["Low"], self.df["Close"])
        return self

    def _add_tsi(self) -> FeatureEngineering:
        # TSI: Momentum oscillator that measures the trend and strength of price movements while smoothing
        # noise using double exponential moving averages.
        self.df["TSI"] = ti.tsi(self.df["Close"])

        # Optional: signal line
        # self.df["TSI_SIGNAL"] = self.df["TSI"].ewm(span=7, adjust=False).mean()
        return self

    def _add_stochastic_oscillator(self) -> FeatureEngineering:
        if self.__n_rows < 14:
            return self
        self.df["Stochastic"] = ti.stochastic_oscillator(self.df["High"], self.df["Low"], self.df["Close"])

        return self

    def _add_bollinger_bands(self) -> FeatureEngineering:
        if self.__n_rows < 20:
            return self
        # BB_Upper / BB_Lower / BB_Middle: Upper/lower bands around 20-day SMA ± 2 standard deviations.
        # Measures volatility.
        # BB_Width: Width between upper and lower band. Higher width = higher volatility.
        # BB_Width_Ratio: BB_Width relative to middle band. Normalized volatility.
        # BB_B: Measures the relative position of the closing price within the Bollinger Bands.
        (
            self.df["BB_Upper"],
            self.df["BB_Lower"],
            self.df["BB_Middle"],
            self.df["BB_Width"],
            self.df["BB_Width_Ratio"],
            self.df["BB_B"],
        ) = ti.bollinger_bands(self.df["Close"])

        return self

    def _add_ichimoku_cloud(self):
        pass

    def _add_price_changes(self) -> FeatureEngineering:
        # Price_Change_nd: Percent change in closing price over n days. Measures short-term returns.
        for days in _price_change_windows:
            if self.__n_rows >= days:
                self.df[f"Price_Change_{days}d"] = ti.price_change(self.df["Close"], days)

        return self

    def _add_volume_indicators(self) -> FeatureEngineering:
        # Volume_Change: Day-to-day percent change in volume.
        self.df["Volume_Change"] = ti.price_change(self.df["Volume"], 1)
        # Volume_MA_n: n-day moving average of volume. Smoother trend of trading activity.
        for w in _volume_change_windows:
            if self.__n_rows >= w:
                self.df[f"Volume_MA_{w}"] = ti.sma(self.df["Volume"], w)
        # Volume_Ratio / Volume_SMA_Ratio: Current volume relative to short/medium-term average. Indicates spikes.
        self.df["Volume_Ratio"] = ti.ratio(self.df["Volume"], self.df.get("Volume_MA_5", pd.Series(1)))
        self.df["Volume_SMA_Ratio"] = ti.ratio(self.df["Volume"], self.df.get("Volume_MA_20", pd.Series(1)))
        # Volume_MA_Ratio_10_20: Ratio of 10-day MA to 20-day MA. Detects accelerating/decelerating trading activity.
        if "Volume_MA_10" in self.df.columns and "Volume_MA_20" in self.df.columns:
            self.df["Volume_MA_Ratio_10_20"] = ti.ratio(self.df["Volume_MA_10"], self.df["Volume_MA_20"])

        return self

    def _add_volatility_indicators(self) -> FeatureEngineering:
        for w in _volatility_windows:
            if self.__n_rows >= w:
                # Volatility_nd: n-day rolling standard deviation of closing price. Measures risk and variability.
                self.df[f"Volatility_{w}d"] = ti.std(self.df["Close"], w)
        # Volatility_Ratio_10_20: Ratio of 10-day to 20-day volatility.
        # Shows recent volatility relative to longer term.
        if "Volatility_10d" in self.df.columns and "Volatility_20d" in self.df.columns:
            self.df["Volatility_Ratio_10_20"] = ti.ratio(self.df["Volatility_10d"], self.df["Volatility_20d"])
        return self

    def _add_high_low_range(self) -> FeatureEngineering:
        # HL_Range: Daily range as % of close. Measures intraday volatility.
        self.df["HL_Range"] = ti.ratio(self.df["High"] - self.df["Low"], self.df["Close"])
        # HL_Range_MA_5 / HL_Range_MA_20: 5/20-day moving average of HL_Range. Smooths daily range trends.
        if self.__n_rows >= 5:
            self.df["HL_Range_MA_5"] = ti.sma(self.df["HL_Range"], 5)
        # HL_Range_Ratio_5_20: Ratio of short-term to medium-term HL range.
        # Signals expansion/contraction in intraday volatility.
        if self.__n_rows >= 20:
            self.df["HL_Range_MA_20"] = ti.sma(self.df["HL_Range"], 20)
            if "HL_Range_MA_5" in self.df.columns:
                self.df["HL_Range_Ratio_5_20"] = ti.ratio(self.df["HL_Range_MA_5"], self.df["HL_Range_MA_20"])

        return self

    def _add_atr(self) -> FeatureEngineering:
        if self.__n_rows < 14:
            return self
        self.df["ATR"] = ti.atr(self.df["High"], self.df["Low"], self.df["Close"])

        return self

    def _add_lagged_returns(self) -> FeatureEngineering:
        # TODO add for similar windows to bellow
        self.df["Return_1d"] = ti.price_change(self.df["Close"], 1)

        # Return_Lag_n: n-day lagged return. Captures momentum or reversal effects from past returns.
        for lag in _lagged_returns_windows:
            self.df[f"Return_Lag_{lag}"] = ti.lagged_return(self.df["Close"], lag)
            # TODO:
            # self.df[f"Lagged_Return_{lag}d_Volatility"] = self.df[f"Lagged_Return_{lag}d"].rolling(20).std()
            # self.df[f"Lagged_Return_{lag}d_Volatility_Ratio"] = (
            #     self.df[f"Lagged_Return_{lag}d_Volatility"] / self.df["Volatility_20d"].replace(0, np.nan)
            # )
            # self.df[f"Lagged_Return_{lag}d_Volatility_Ratio_SMA_20"] = (
            #     self.df[f"Lagged_Return_{lag}d_Volatility_Ratio"].rolling(20).mean()
            # )

        return self

    def _add_momentum(self) -> FeatureEngineering:
        for m in [1, 5, 10, 20, 50]:
            if self.__n_rows >= m:
                # Momentum_n: Price change over n days. Shows trend strength.
                self.df[f"Momentum_{m}"] = ti.momentum(self.df["Close"], m)
        # Momentum_Ratio_5_10 / Momentum_Ratio_10_20: Ratio of short-term to medium-term momentum.
        # Highlights trend acceleration/deceleration.
        if "Momentum_5" in self.df.columns and "Momentum_10" in self.df.columns:
            self.df["Momentum_Ratio_5_10"] = ti.ratio(self.df["Momentum_5"], self.df["Momentum_10"])
        if "Momentum_10" in self.df.columns and "Momentum_20" in self.df.columns:
            self.df["Momentum_Ratio_10_20"] = ti.ratio(self.df["Momentum_10"], self.df["Momentum_20"])

        return self

    def _add_rate_of_change(self) -> FeatureEngineering:
        # ROC_n: Percent change in price over n days. Momentum indicator.
        for r in _rate_of_change_windows:
            if self.__n_rows >= r:
                self.df[f"ROC_{r}"] = ti.rate_of_change(self.df["Close"], r)

        return self

    def _add_directional_indicators(self) -> FeatureEngineering:
        if "ATR" in self.df.columns and self.__n_rows >= 14:
            # Plus_DI / Minus_DI: 14-day directional indicators. Show trend direction strength.
            # ADX: Average Directional Index. Measures trend strength regardless of direction.
            self.df["Plus_DI"], self.df["Minus_DI"], self.df["ADX"] = ti.directional_indicators(
                self.df["High"], self.df["Low"], self.df["ATR"]
            )

            # TODO Maybe
            # self.df["ATR_Directional_Up"] = (self.df["High"] - self.df["Close"].shift(1)).where(
            #     self.df["High"] - self.df["Close"].shift(1) > self.df["ATR"], 0
            # )
            # self.df["ATR_Directional_Down"] = (self.df["Close"].shift(1) - self.df["Low"]).where(
            #     self.df["Close"].shift(1) - self.df["Low"] > self.df["ATR"], 0
            # )
            # self.df["ATR_Directional_Up_Ratio"] = self.df["ATR_Directional_Up"] / self.df["ATR"].replace(0, np.nan)
            # self.df["ATR_Directional_Down_Ratio"] = (
            #     self.df["ATR_Directional_Down"] / self.df["ATR"].replace(0, np.nan)
            # )
        return self

    def _add_sharpe(self) -> FeatureEngineering:
        # Sharpe Ratio in Windows
        for w in _sharpe_windows:
            self.df[f"Sharpe_{w}"] = ti.sharpe(self.df["Return_1d"], w)
        return self

    def _add_regime_features(self) -> FeatureEngineering:
        returns = self.df["Close"].pct_change(1).fillna(0)
        regime_df = classify_regime(self.df["Close"], returns)
        for col in regime_df.columns:
            self.df[col] = regime_df[col]
        return self

    def _add_candlestick_patterns(self) -> FeatureEngineering:
        self.df["Bull_Engulfing"], self.df["Doji"] = ti.candlestick_patterns(
            self.df["Open"], self.df["Close"], self.df["High"], self.df["Low"]
        )

        return self

    def add_all_technical_indicators(self):
        if self.__calculated:
            return
        # Calculate technical indicators
        (
            self._add_moving_averages()
            ._add_safe_price_ratios()
            ._add_macd()
            ._add_rsi()
            ._add_cci()
            ._add_williams_r()
            ._add_stochastic_oscillator()
            ._add_bollinger_bands()
            ._add_price_changes()
            ._add_volume_indicators()
            ._add_volatility_indicators()
            ._add_high_low_range()
            ._add_atr()
            ._add_lagged_returns()
            ._add_momentum()
            ._add_rate_of_change()
            ._add_directional_indicators()
            ._add_sharpe()
            ._add_candlestick_patterns()
            ._add_regime_features()
        )

        # Shift all features by 1 day
        feature_columns = self.df.columns.difference(_all_cols)
        self.df[feature_columns] = self.df[feature_columns].shift(1)

        # Cleanup - Replace inf with NaN, then fill or drop
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.df.ffill(inplace=True)
        self.df.bfill(inplace=True)

        self.__calculated = True

    def _create_target(self, target_days: int):
        self.df["Target"] = self.df["Close"].pct_change(target_days).shift(-target_days)
        return self.df

    def prepare_features(self, scale: bool = False):
        if self.__df_prepared is not None:
            return self.__df_prepared
        df = self.df.copy()
        feature_columns = df.columns.difference(_all_cols_with_targets)
        # Fill missing values instead of dropping all
        df[feature_columns] = df[feature_columns].fillna(0)

        # If Target has NaN at the end, drop only those rows
        df_clean = df.dropna(subset=["Target"])
        if len(df_clean) == 0:
            raise ValueError("No valid data after processing features")

        x = df_clean[feature_columns]
        y = df_clean["Target"]
        dates = df_clean.index
        prices = df_clean["Close"]

        # Scale features if requested
        if scale:
            scaler = StandardScaler()
            cat_cols = ["ticker", "sector", "industry"]
            num_cols = [c for c in x.columns if c not in cat_cols]
            x = x.copy()
            x[num_cols] = scaler.fit_transform(x[num_cols])

        self.__df_prepared = x, y, dates, prices, feature_columns.tolist()  # type: ignore[assignment]
        return self.__df_prepared

    def create_target_features(self, target_days: int = 10):
        self.add_all_technical_indicators()
        return self._create_target(target_days=target_days)
