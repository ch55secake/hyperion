import numpy as np
import pandas as pd


class FeatureEngineering:
    """Creates technical indicators and features for machine learning"""

    @staticmethod
    def add_technical_indicators(df):
        """Add various technical indicators to the dataframe"""
        df = df.copy()

        # Simple Moving Averages
        df["SMA_5"] = df["Close"].rolling(window=5).mean()
        df["SMA_10"] = df["Close"].rolling(window=10).mean()
        df["SMA_20"] = df["Close"].rolling(window=20).mean()
        df["SMA_50"] = df["Close"].rolling(window=50).mean()

        # Exponential Moving Averages
        df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
        df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()

        # MACD
        df["MACD"] = df["EMA_12"] - df["EMA_26"]
        df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
        df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

        # RSI
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        df["BB_Middle"] = df["Close"].rolling(window=20).mean()
        bb_std = df["Close"].rolling(window=20).std()
        df["BB_Upper"] = df["BB_Middle"] + (bb_std * 2)
        df["BB_Lower"] = df["BB_Middle"] - (bb_std * 2)
        df["BB_Width"] = df["BB_Upper"] - df["BB_Lower"]

        # Price changes
        df["Price_Change_1d"] = df["Close"].pct_change(1)
        df["Price_Change_5d"] = df["Close"].pct_change(5)
        df["Price_Change_10d"] = df["Close"].pct_change(10)

        # Volume indicators
        df["Volume_Change"] = df["Volume"].pct_change(1)
        df["Volume_MA_5"] = df["Volume"].rolling(window=5).mean()
        df["Volume_Ratio"] = df["Volume"] / df["Volume_MA_5"]

        # Volatility
        df["Volatility_10d"] = df["Close"].rolling(window=10).std()
        df["Volatility_20d"] = df["Close"].rolling(window=20).std()

        # High-Low range
        df["HL_Range"] = (df["High"] - df["Low"]) / df["Close"]
        df["HL_Range_MA"] = df["HL_Range"].rolling(window=5).mean()

        # Average True Range (ATR)
        high_low = df["High"] - df["Low"]
        high_close = np.abs(df["High"] - df["Close"].shift())
        low_close = np.abs(df["Low"] - df["Close"].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df["ATR"] = true_range.rolling(14).mean()

        # Additional predictive features
        # Lagged returns (previous days' returns as features)
        for lag in [1, 2, 3, 5, 10]:
            df[f"Return_Lag_{lag}"] = df["Close"].pct_change(1).shift(lag)

        # Price momentum indicators
        df["Momentum_5"] = df["Close"] - df["Close"].shift(5)
        df["Momentum_10"] = df["Close"] - df["Close"].shift(10)
        df["Momentum_20"] = df["Close"] - df["Close"].shift(20)

        # Price relative to moving averages
        df["Price_to_SMA20"] = df["Close"] / df["SMA_20"]
        df["Price_to_SMA50"] = df["Close"] / df["SMA_50"]

        # Volatility ratios
        df["Volatility_Ratio"] = df["Volatility_10d"] / df["Volatility_20d"]

        # Volume trends
        df["Volume_SMA_Ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()

        # Rate of change
        df["ROC_5"] = ((df["Close"] - df["Close"].shift(5)) / df["Close"].shift(5)) * 100
        df["ROC_10"] = ((df["Close"] - df["Close"].shift(10)) / df["Close"].shift(10)) * 100

        # Stochastic Oscillator
        low_14 = df["Low"].rolling(window=14).min()
        high_14 = df["High"].rolling(window=14).max()
        df["Stochastic"] = 100 * (df["Close"] - low_14) / (high_14 - low_14)

        # MACD momentum
        df["MACD_Momentum"] = df["MACD"].diff()

        return df

    @staticmethod
    def create_target(df, target_days=1):
        """Create target variable (future returns)"""
        df = df.copy()
        df["Target"] = df["Close"].shift(-target_days) / df["Close"] - 1
        return df

    @staticmethod
    def prepare_features(df):
        """Prepare feature matrix and target vector"""
        feature_columns = [
            # Moving averages
            "SMA_5",
            "SMA_10",
            "SMA_20",
            "SMA_50",
            "EMA_12",
            "EMA_26",
            # MACD indicators
            "MACD",
            "MACD_Signal",
            "MACD_Hist",
            "MACD_Momentum",
            # Momentum and oscillators
            "RSI",
            "Stochastic",
            # Bollinger Bands
            "BB_Width",
            # Price changes
            "Price_Change_1d",
            "Price_Change_5d",
            "Price_Change_10d",
            # Volume indicators
            "Volume_Change",
            "Volume_Ratio",
            "Volume_SMA_Ratio",
            # Volatility
            "Volatility_10d",
            "Volatility_20d",
            "Volatility_Ratio",
            # Other indicators
            "HL_Range",
            "ATR",
            # Lagged returns
            "Return_Lag_1",
            "Return_Lag_2",
            "Return_Lag_3",
            "Return_Lag_5",
            "Return_Lag_10",
            # Momentum
            "Momentum_5",
            "Momentum_10",
            "Momentum_20",
            # Price ratios
            "Price_to_SMA20",
            "Price_to_SMA50",
            # Rate of change
            "ROC_5",
            "ROC_10",
        ]

        # Remove rows with NaN values
        df_clean = df.dropna()

        if len(df_clean) == 0:
            raise ValueError("No valid data after removing NaN values")

        x = df_clean[feature_columns]
        y = df_clean["Target"]
        dates = df_clean.index
        prices = df_clean["Close"]

        return x, y, dates, prices, feature_columns


def create_target_features(df):
    df_features = FeatureEngineering.add_technical_indicators(df)
    return FeatureEngineering.create_target(df_features, target_days=1)
