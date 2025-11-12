from __future__ import annotations

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler

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


class FeatureEngineering:
    """Creates technical indicators and features for machine learning, robust to small/incomplete data."""

    def __init__(self, df: DataFrame):
        self.df: DataFrame = self._ensure_required_columns(df)
        self.__n_rows: int = len(self.df)
        self.__calculated: bool= False
        self.__df_prepared: DataFrame | None = None

    @staticmethod
    def _ensure_required_columns(df: DataFrame) -> DataFrame:
        for col in _required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        return df.copy()

    def get_df(self) -> DataFrame:
        return self.df

    def _add_moving_averages(self) -> FeatureEngineering:
        for w in _ma_windows:
            if self.__n_rows >= w:
                self.df[f"SMA_{w}"] = self.df["Close"].rolling(window=w).mean()
                self.df[f"EMA_{w}"] = self.df["Close"].ewm(span=w, adjust=False).mean()

        return self

    def _add_safe_price_ratios(self) -> FeatureEngineering:
        self.df["Price_to_SMA20"] = self.df["Close"] / self.df.get("SMA_20", pd.Series(1)).replace(0, np.nan)
        self.df["Price_to_SMA50"] = self.df["Close"] / self.df.get("SMA_50", pd.Series(1)).replace(0, np.nan)
        self.df["Price_to_EMA12"] = self.df["Close"] / self.df.get("EMA_12", pd.Series(1)).replace(0, np.nan)
        self.df["Price_to_EMA26"] = self.df["Close"] / self.df.get("EMA_26", pd.Series(1)).replace(0, np.nan)
        if "EMA_12" in self.df.columns and "EMA_26" in self.df.columns:
            self.df["EMA_12_26_Ratio"] = self.df["EMA_12"] / self.df["EMA_26"].replace(0, np.nan)
        if "SMA_5" in self.df.columns and "SMA_20" in self.df.columns:
            self.df["SMA_5_20_Ratio"] = self.df["SMA_5"] / self.df["SMA_20"].replace(0, np.nan)
        if "SMA_10" in self.df.columns and "SMA_50" in self.df.columns:
            self.df["SMA_10_50_Ratio"] = self.df["SMA_10"] / self.df["SMA_50"].replace(0, np.nan)

        return self

    def _add_macd(self) -> FeatureEngineering:
        if "EMA_12" in self.df.columns and "EMA_26" in self.df.columns:
            self.df["MACD"] = self.df["EMA_12"] - self.df["EMA_26"]
            self.df["MACD_Signal"] = self.df["MACD"].ewm(span=9, adjust=False).mean()
            self.df["MACD_Hist"] = self.df["MACD"] - self.df["MACD_Signal"]
            self.df["MACD_Momentum"] = self.df["MACD"].diff()
            self.df["MACD_Cross"] = (self.df["MACD"] > self.df["MACD_Signal"]).astype(int)

        return self

    def _add_rsi(self) -> FeatureEngineering:
        if self.__n_rows < 14:
            return
        delta = self.df["Close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        self.df["RSI"] = 100 - (100 / (1 + rs))
        self.df["RSI_Overbought"] = (self.df["RSI"] > 70).astype(int)
        self.df["RSI_Oversold"] = (self.df["RSI"] < 30).astype(int)

        return self

    def _add_stochastic_oscillator(self) -> FeatureEngineering:
        if self.__n_rows < 14:
            return self
        low_14 = self.df["Low"].rolling(14).min()
        high_14 = self.df["High"].rolling(14).max()
        self.df["Stochastic"] = 100 * (self.df["Close"] - low_14) / (high_14 - low_14).replace(0, np.nan)

        return self

    def _add_bollinger_bands(self) -> FeatureEngineering:
        if self.__n_rows < 20:
            return self
        self.df["BB_Upper"] = self.df["Close"].rolling(20).mean() + 2 * self.df["Close"].rolling(20).std()
        self.df["BB_Lower"] = self.df["Close"].rolling(20).mean() - 2 * self.df["Close"].rolling(20).std()
        self.df["BB_Middle"] = self.df["Close"].rolling(20).mean()
        self.df["BB_Width"] = self.df["BB_Upper"] - self.df["BB_Lower"]
        self.df["BB_Width_Ratio"] = self.df["BB_Width"] / self.df["BB_Middle"].replace(0, np.nan)
        self.df["Price_BB_Position"] = (self.df["Close"] - self.df["BB_Lower"]) / (
            self.df["BB_Upper"] - self.df["BB_Lower"]
        ).replace(0, np.nan)

        return self

    def _add_price_changes(self) -> FeatureEngineering:
        for days in _price_change_windows:
            if self.__n_rows >= days:
                self.df[f"Price_Change_{days}d"] = self.df["Close"].pct_change(days)

        return self

    def _add_volume_indicators(self) -> FeatureEngineering:
        self.df["Volume_Change"] = self.df["Volume"].pct_change(1)
        for w in _volume_change_windows:
            if self.__n_rows >= w:
                self.df[f"Volume_MA_{w}"] = self.df["Volume"].rolling(window=w).mean()
        self.df["Volume_Ratio"] = self.df["Volume"] / self.df.get("Volume_MA_5", pd.Series(1)).replace(0, np.nan)
        self.df["Volume_SMA_Ratio"] = self.df["Volume"] / self.df.get("Volume_MA_20", pd.Series(1)).replace(0, np.nan)
        if "Volume_MA_10" in self.df.columns and "Volume_MA_20" in self.df.columns:
            self.df["Volume_MA_Ratio_10_20"] = self.df["Volume_MA_10"] / self.df["Volume_MA_20"].replace(0, np.nan)

        return self

    def _add_volatility_indicators(self) -> FeatureEngineering:
        for w in _volatility_windows:
            if self.__n_rows >= w:
                self.df[f"Volatility_{w}d"] = self.df["Close"].rolling(window=w).std()
        if "Volatility_10d" in self.df.columns and "Volatility_20d" in self.df.columns:
            self.df["Volatility_Ratio_10_20"] = self.df["Volatility_10d"] / self.df["Volatility_20d"].replace(0, np.nan)

        return self

    def _add_high_low_range(self) -> FeatureEngineering:
        self.df["HL_Range"] = (self.df["High"] - self.df["Low"]) / self.df["Close"].replace(0, np.nan)
        if self.__n_rows >= 5:
            self.df["HL_Range_MA_5"] = self.df["HL_Range"].rolling(5).mean()
        if self.__n_rows >= 20:
            self.df["HL_Range_MA_20"] = self.df["HL_Range"].rolling(20).mean()
            if "HL_Range_MA_5" in self.df.columns:
                self.df["HL_Range_Ratio_5_20"] = self.df["HL_Range_MA_5"] / self.df["HL_Range_MA_20"].replace(0, np.nan)

        return self

    def _add_atr(self) -> FeatureEngineering:
        if self.__n_rows < 14:
            return self
        high_low = self.df["High"] - self.df["Low"]
        high_close = np.abs(self.df["High"] - self.df["Close"].shift())
        low_close = np.abs(self.df["Low"] - self.df["Close"].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        self.df["ATR"] = true_range.rolling(14).mean()

        return self

    def _add_lagged_returns(self) -> FeatureEngineering:
        for lag in _lagged_returns_windows:
            self.df[f"Return_Lag_{lag}"] = self.df["Close"].pct_change(1).shift(lag)
            # TODO:
            # self.df[f"Lagged_Return_{lag}d_Volatility"] = self.df[f"Lagged_Return_{lag}d"].rolling(20).std()
            # self.df[f"Lagged_Return_{lag}d_Volatility_Ratio"] = self.df[f"Lagged_Return_{lag}d_Volatility"] / self.df["Volatility_20d"].replace(0, np.nan)
            # self.df[f"Lagged_Return_{lag}d_Volatility_Ratio_SMA_20"] = self.df[f"Lagged_Return_{lag}d_Volatility_Ratio"].rolling(20).mean()

        return self

    def _add_momentum(self) -> FeatureEngineering:
        for m in [1, 5, 10, 20, 50]:
            if self.__n_rows >= m:
                self.df[f"Momentum_{m}"] = self.df["Close"] - self.df["Close"].shift(m)
            if "Momentum_5" in self.df.columns and "Momentum_10" in self.df.columns:
                self.df["Momentum_Ratio_5_10"] = self.df["Momentum_5"] / self.df["Momentum_10"].replace(0, np.nan)
            if "Momentum_10" in self.df.columns and "Momentum_20" in self.df.columns:
                self.df["Momentum_Ratio_10_20"] = self.df["Momentum_10"] / self.df["Momentum_20"].replace(0, np.nan)

        return self

    def _add_rate_of_change(self) -> FeatureEngineering:
        for r in _rate_of_change_windows:
            if self.__n_rows >= r:
                self.df[f"ROC_{r}"] = (
                    (self.df["Close"] - self.df["Close"].shift(r)) / self.df["Close"].shift(r)
                ).replace(0, np.nan) * 100

        return self

    def _add_directional_indicators(self) -> FeatureEngineering:
        if "ATR" in self.df.columns and self.__n_rows >= 14:
            high_diff = self.df["High"].diff()
            low_diff = -self.df["Low"].diff()
            plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0.0)
            minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0.0)
            trur = self.df["ATR"]
            self.df["Plus_DI"] = 100 * (plus_dm.rolling(14).sum() / trur.replace(0, np.nan))
            self.df["Minus_DI"] = 100 * (minus_dm.rolling(14).sum() / trur.replace(0, np.nan))
            self.df["ADX"] = 100 * (
                abs(self.df["Plus_DI"] - self.df["Minus_DI"])
                / (self.df["Plus_DI"] + self.df["Minus_DI"]).replace(0, np.nan)
            )
            # TODO Maybe
            # self.df["ATR_Directional_Up"] = (self.df["High"] - self.df["Close"].shift(1)).where(self.df["High"] - self.df["Close"].shift(1) > self.df["ATR"], 0)
            # self.df["ATR_Directional_Down"] = (self.df["Close"].shift(1) - self.df["Low"]).where(self.df["Close"].shift(1) - self.df["Low"] > self.df["ATR"], 0)
            # self.df["ATR_Directional_Up_Ratio"] = self.df["ATR_Directional_Up"] / self.df["ATR"].replace(0, np.nan)
            # self.df["ATR_Directional_Down_Ratio"] = self.df["ATR_Directional_Down"] / self.df["ATR"].replace(0, np.nan)
        return self

    def _add_candlestick_patterns(self) -> FeatureEngineering:
        self.df['Bull_Engulfing'] = ((self.df['Close'] > self.df['Open'].shift(1)) & (self.df['Open'] < self.df['Close'].shift(1))).astype(int)
        self.df['Doji'] = (abs(self.df['Close'] - self.df['Open']) / (self.df['High'] - self.df['Low']) < 0.1).astype(int)

        return self


    def add_all_technical_indicators(self):
        if self.__calculated:
            return
        # Calculate technical indicators
        (self._add_moving_averages()
         ._add_safe_price_ratios()
         ._add_macd()
         ._add_rsi()
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
         ._add_candlestick_patterns())

        # Shift all features by 1 day
        feature_columns = self.df.columns.difference(_all_cols)
        self.df[feature_columns] = self.df[feature_columns].shift(1)

        # Cleanup - Replace inf with NaN, then fill or drop
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.df.fillna(method="ffill", inplace=True)
        self.df.fillna(method="bfill", inplace=True)

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
            x = pd.DataFrame(scaler.fit_transform(x), columns=x.columns, index=x.index)

        self.__df_prepared = x, y, dates, prices, feature_columns.tolist()
        return self.__df_prepared

    def create_target_features(self, target_days: int = 10):
        self.add_all_technical_indicators()
        return self._create_target(target_days=target_days)
