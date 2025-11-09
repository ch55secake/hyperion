import numpy as np
import pandas as pd


class FeatureEngineering:
    """Creates technical indicators and features for machine learning, robust to small/incomplete data."""

    @staticmethod
    def add_technical_indicators(df):
        df = df.copy()

        # Ensure required columns exist
        required_cols = ["Close", "High", "Low", "Volume"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        n_rows = len(df)

        # ============================================================
        # SIMPLE AND EXPONENTIAL MOVING AVERAGES
        # ============================================================
        ma_windows = [5, 10, 20, 50, 100, 200]
        for w in ma_windows:
            if n_rows >= w:
                df[f"SMA_{w}"] = df["Close"].rolling(window=w).mean()
                df[f"EMA_{w}"] = df["Close"].ewm(span=w, adjust=False).mean()

        # Safe price ratios
        df["Price_to_SMA20"] = df["Close"] / df.get("SMA_20", pd.Series(1)).replace(0, np.nan)
        df["Price_to_SMA50"] = df["Close"] / df.get("SMA_50", pd.Series(1)).replace(0, np.nan)
        df["Price_to_EMA12"] = df["Close"] / df.get("EMA_12", pd.Series(1)).replace(0, np.nan)
        df["Price_to_EMA26"] = df["Close"] / df.get("EMA_26", pd.Series(1)).replace(0, np.nan)
        if "EMA_12" in df.columns and "EMA_26" in df.columns:
            df["EMA_12_26_Ratio"] = df["EMA_12"] / df["EMA_26"].replace(0, np.nan)
        if "SMA_5" in df.columns and "SMA_20" in df.columns:
            df["SMA_5_20_Ratio"] = df["SMA_5"] / df["SMA_20"].replace(0, np.nan)
        if "SMA_10" in df.columns and "SMA_50" in df.columns:
            df["SMA_10_50_Ratio"] = df["SMA_10"] / df["SMA_50"].replace(0, np.nan)

        # ============================================================
        # MACD
        # ============================================================
        if "EMA_12" in df.columns and "EMA_26" in df.columns:
            df["MACD"] = df["EMA_12"] - df["EMA_26"]
            df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
            df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
            df["MACD_Momentum"] = df["MACD"].diff()
            df["MACD_Cross"] = (df["MACD"] > df["MACD_Signal"]).astype(int)

        # ============================================================
        # RSI
        # ============================================================
        if n_rows >= 14:
            delta = df["Close"].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = -delta.where(delta < 0, 0).rolling(14).mean()
            rs = gain / loss.replace(0, np.nan)
            df["RSI"] = 100 - (100 / (1 + rs))
            df["RSI_Overbought"] = (df["RSI"] > 70).astype(int)
            df["RSI_Oversold"] = (df["RSI"] < 30).astype(int)

        # ============================================================
        # STOCHASTIC OSCILLATOR
        # ============================================================
        if n_rows >= 14:
            low_14 = df["Low"].rolling(14).min()
            high_14 = df["High"].rolling(14).max()
            df["Stochastic"] = 100 * (df["Close"] - low_14) / (high_14 - low_14).replace(0, np.nan)

        # ============================================================
        # BOLLINGER BANDS
        # ============================================================
        if n_rows >= 20:
            df["BB_Middle"] = df["Close"].rolling(20).mean()
            bb_std = df["Close"].rolling(20).std()
            df["BB_Upper"] = df["BB_Middle"] + 2 * bb_std
            df["BB_Lower"] = df["BB_Middle"] - 2 * bb_std
            df["BB_Width"] = df["BB_Upper"] - df["BB_Lower"]
            df["BB_Width_Ratio"] = df["BB_Width"] / df["BB_Middle"].replace(0, np.nan)
            df["Price_BB_Position"] = (df["Close"] - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"]).replace(
                0, np.nan
            )

        # ============================================================
        # PRICE CHANGES
        # ============================================================
        for days in [1, 5, 10, 20, 50, 100]:
            if n_rows >= days:
                df[f"Price_Change_{days}d"] = df["Close"].pct_change(days)

        # ============================================================
        # VOLUME INDICATORS
        # ============================================================
        df["Volume_Change"] = df["Volume"].pct_change(1)
        for w in [5, 10, 20, 50]:
            if n_rows >= w:
                df[f"Volume_MA_{w}"] = df["Volume"].rolling(window=w).mean()
        df["Volume_Ratio"] = df["Volume"] / df.get("Volume_MA_5", pd.Series(1)).replace(0, np.nan)
        df["Volume_SMA_Ratio"] = df["Volume"] / df.get("Volume_MA_20", pd.Series(1)).replace(0, np.nan)
        if "Volume_MA_10" in df.columns and "Volume_MA_20" in df.columns:
            df["Volume_MA_Ratio_10_20"] = df["Volume_MA_10"] / df["Volume_MA_20"].replace(0, np.nan)

        # ============================================================
        # VOLATILITY
        # ============================================================
        for w in [5, 10, 20, 50]:
            if n_rows >= w:
                df[f"Volatility_{w}d"] = df["Close"].rolling(window=w).std()
        if "Volatility_10d" in df.columns and "Volatility_20d" in df.columns:
            df["Volatility_Ratio_10_20"] = df["Volatility_10d"] / df["Volatility_20d"].replace(0, np.nan)

        # ============================================================
        # HIGH-LOW RANGE
        # ============================================================
        df["HL_Range"] = (df["High"] - df["Low"]) / df["Close"].replace(0, np.nan)
        if n_rows >= 5:
            df["HL_Range_MA_5"] = df["HL_Range"].rolling(5).mean()
        if n_rows >= 20:
            df["HL_Range_MA_20"] = df["HL_Range"].rolling(20).mean()
            if "HL_Range_MA_5" in df.columns:
                df["HL_Range_Ratio_5_20"] = df["HL_Range_MA_5"] / df["HL_Range_MA_20"].replace(0, np.nan)

        # ============================================================
        # ATR
        # ============================================================
        high_low = df["High"] - df["Low"]
        high_close = np.abs(df["High"] - df["Close"].shift())
        low_close = np.abs(df["Low"] - df["Close"].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        if n_rows >= 14:
            df["ATR"] = true_range.rolling(14).mean()

        # ============================================================
        # LAGGED RETURNS
        # ============================================================
        for lag in [1, 2, 3, 5, 10, 20]:
            df[f"Return_Lag_{lag}"] = df["Close"].pct_change(1).shift(lag)

        # ============================================================
        # MOMENTUM
        # ============================================================
        for m in [5, 10, 20, 50]:
            if n_rows >= m:
                df[f"Momentum_{m}"] = df["Close"] - df["Close"].shift(m)
        if "Momentum_5" in df.columns and "Momentum_10" in df.columns:
            df["Momentum_Ratio_5_10"] = df["Momentum_5"] / df["Momentum_10"].replace(0, np.nan)
        if "Momentum_10" in df.columns and "Momentum_20" in df.columns:
            df["Momentum_Ratio_10_20"] = df["Momentum_10"] / df["Momentum_20"].replace(0, np.nan)

        # ============================================================
        # RATE OF CHANGE
        # ============================================================
        for r in [5, 10, 20, 50]:
            if n_rows >= r:
                df[f"ROC_{r}"] = ((df["Close"] - df["Close"].shift(r)) / df["Close"].shift(r)).replace(0, np.nan) * 100

        # ============================================================
        # ADX / DIRECTIONAL INDICATORS
        # ============================================================
        if "ATR" in df.columns and n_rows >= 14:
            high_diff = df["High"].diff()
            low_diff = -df["Low"].diff()
            plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0.0)
            minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0.0)
            trur = df["ATR"]
            df["Plus_DI"] = 100 * (plus_dm.rolling(14).sum() / trur.replace(0, np.nan))
            df["Minus_DI"] = 100 * (minus_dm.rolling(14).sum() / trur.replace(0, np.nan))
            df["ADX"] = 100 * (
                abs(df["Plus_DI"] - df["Minus_DI"]) / (df["Plus_DI"] + df["Minus_DI"]).replace(0, np.nan)
            )

        # ============================================================
        # SHIFT ALL FEATURES BY 1 DAY
        # ============================================================
        feature_columns = df.columns.difference(["Open", "High", "Low", "Close", "Volume"])
        df[feature_columns] = df[feature_columns].shift(1)

        # ============================================================
        # FINAL CLEANUP
        # Replace inf with NaN, then fill or drop
        # ============================================================
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(method="ffill", inplace=True)
        df.fillna(method="bfill", inplace=True)

        return df

    @staticmethod
    def create_target(df, target_days=1):
        df = df.copy()
        df["Target"] = df["Close"].pct_change(target_days).shift(-target_days)
        return df

    @staticmethod
    def prepare_features(df):
        feature_columns = df.columns.difference(["Open", "High", "Low", "Close", "Volume", "Target"])

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
        return x, y, dates, prices, feature_columns.tolist()


def create_target_features(df, target_days=10):
    df_features = FeatureEngineering.add_technical_indicators(df)
    return FeatureEngineering.create_target(df_features, target_days=target_days)
