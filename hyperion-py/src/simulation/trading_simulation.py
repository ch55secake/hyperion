from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
from pandas import DataFrame, Series

from src.feature import FeatureEngineering
from src.visualisation import Visualizer
from src.xbg.xgb_predictor import XGBoostStockPredictor


class TradingSimulator:
    """
    Simulates trading based on model predictions
    """

    def __init__(self, initial_capital=10000, transaction_cost=0.001):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.portfolio_history = []
        self.trades = []

    def simulate(
        self,
        predictions,
        actual_returns,
        prices=None,
        dates=None,
        threshold="auto",
        strategy="directional",
        use_returns=True,
    ):
        """
        Simulate trading strategy and evaluate predictive performance.
        """
        print("\n" + "=" * 60)
        print("Running Trading Simulation")
        print("=" * 60)

        pred_array = np.array(predictions)

        # --- Determine Threshold ---
        if threshold == "auto":
            threshold = np.percentile(np.abs(pred_array), 25)
            print(f"\nAuto threshold (25th percentile): {threshold:.6f}")
        elif threshold == "adaptive":
            threshold = 0.3 * np.std(pred_array)
            print(f"\nAdaptive threshold (0.3 std): {threshold:.6f}")
        else:
            threshold = float(threshold)
            print(f"\nFixed threshold: {threshold:.6f}")

        # --- Initialize Portfolio ---
        capital = self.initial_capital
        shares = 0
        position = None
        entry_price = 0
        hold_counter = 0

        for i, pred_return in enumerate(predictions):
            actual_return = actual_returns.iloc[i] if hasattr(actual_returns, "iloc") else actual_returns[i]
            date = dates.iloc[i] if hasattr(dates, "iloc") else (dates[i] if dates is not None else i)

            # Determine current price
            if use_returns:
                current_price = prices.iloc[i] * (1 + actual_return)
            else:
                current_price = prices.iloc[i] if hasattr(prices, "iloc") else prices[i]

            # --- Trade Logic ---
            if strategy == "directional":
                if position is None and pred_return > 0:
                    # Buy
                    if use_returns:
                        shares = capital
                        capital = 0
                    else:
                        shares = (capital * (1 - self.transaction_cost)) / current_price
                        entry_price = current_price
                        capital = 0
                    position = "long"
                    self.trades.append(
                        {"date": date, "action": "BUY", "predicted_return": pred_return, "price": current_price}
                    )

                elif position == "long" and pred_return <= 0:
                    # Sell
                    if use_returns:
                        capital = shares * (1 + actual_return)
                    else:
                        capital = shares * current_price * (1 - self.transaction_cost)
                    profit = capital - self.initial_capital
                    pnl = actual_return * 100 if use_returns else ((current_price - entry_price) / entry_price) * 100
                    self.trades.append(
                        {
                            "date": date,
                            "action": "SELL",
                            "predicted_return": pred_return,
                            "profit": profit,
                            "pnl_pct": pnl,
                            "price": current_price,
                        }
                    )
                    shares = 0
                    position = None

            elif strategy == "threshold":
                if position is None:
                    if pred_return > threshold:
                        if use_returns:
                            shares = capital
                            capital = 0
                        else:
                            shares = (capital * (1 - self.transaction_cost)) / current_price
                            entry_price = current_price
                            capital = 0
                        position = "long"
                        self.trades.append(
                            {"date": date, "action": "BUY", "predicted_return": pred_return, "price": current_price}
                        )
                else:
                    if pred_return < -threshold:
                        if use_returns:
                            capital = shares * (1 + actual_return)
                        else:
                            capital = shares * current_price * (1 - self.transaction_cost)
                        profit = capital - self.initial_capital
                        pnl = (
                            actual_return * 100 if use_returns else ((current_price - entry_price) / entry_price) * 100
                        )
                        self.trades.append(
                            {
                                "date": date,
                                "action": "SELL",
                                "predicted_return": pred_return,
                                "profit": profit,
                                "pnl_pct": pnl,
                                "price": current_price,
                            }
                        )
                        shares = 0
                        position = None

            elif strategy == "hold_days":
                if position is None and pred_return > threshold:
                    if use_returns:
                        shares = capital
                        capital = 0
                    else:
                        shares = (capital * (1 - self.transaction_cost)) / current_price
                        entry_price = current_price
                        capital = 0
                    position = "long"
                    hold_counter = 0
                    self.trades.append(
                        {"date": date, "action": "BUY", "predicted_return": pred_return, "price": current_price}
                    )
                elif position == "long":
                    hold_counter += 1
                    if hold_counter >= 5 or pred_return < -threshold:
                        if use_returns:
                            capital = shares * (1 + actual_return)
                        else:
                            capital = shares * current_price * (1 - self.transaction_cost)
                        profit = capital - self.initial_capital
                        pnl = (
                            actual_return * 100 if use_returns else ((current_price - entry_price) / entry_price) * 100
                        )
                        self.trades.append(
                            {
                                "date": date,
                                "action": "SELL",
                                "predicted_return": pred_return,
                                "profit": profit,
                                "pnl_pct": pnl,
                                "price": current_price,
                            }
                        )
                        shares = 0
                        position = None

            # --- Portfolio Tracking ---
            if use_returns:
                portfolio_value = capital + (shares if position == "long" else 0)
            else:
                portfolio_value = shares * current_price if position == "long" else capital

            self.portfolio_history.append(
                {
                    "date": date,
                    "portfolio_value": portfolio_value,
                    "position": position if position else "cash",
                    "prediction": pred_return,
                    "price": current_price,  # ✓ Fixed: Add price
                }
            )

        # --- Close remaining position ---
        if position is not None:
            if use_returns:
                final_return = actual_returns.iloc[-1] if hasattr(actual_returns, "iloc") else actual_returns[-1]
                capital = shares * (1 + final_return)
                final_price = current_price * (1 + final_return)
            else:
                final_price = prices.iloc[-1] if hasattr(prices, "iloc") else prices[-1]
                capital = shares * final_price * (1 - self.transaction_cost)

            profit = capital - self.initial_capital
            pnl = (
                np.mean(actual_returns[-5:]) * 100 if use_returns else ((final_price - entry_price) / entry_price) * 100
            )

            if hasattr(dates, "__getitem__"):
                final_date = dates[-1]
            else:
                final_date = None

            self.trades.append(
                {
                    "date": final_date,
                    "action": "SELL (Final)",
                    "profit": profit,
                    "pnl_pct": pnl,
                    "price": final_price,  # ✓ Fixed: Add price
                }
            )

            shares = 0

            # --- Results ---
        final_value = capital
        total_return = (final_value - self.initial_capital) / self.initial_capital
        buy_hold_return = None
        if not use_returns and prices is not None:
            first_price, last_price = prices.iloc[0], prices.iloc[-1]
            buy_hold_return = (last_price - first_price) / first_price

        print("\n" + "=" * 60)
        print("Trading Simulation Results")
        print("=" * 60)
        print(f"Initial Capital:       ${self.initial_capital:,.2f}")
        print(f"Final Portfolio Value: ${final_value:,.2f}")
        print(f"Total Return:          {total_return * 100:.2f}%")
        if buy_hold_return is not None:
            print(f"Buy & Hold Return:     {buy_hold_return * 100:.2f}%")
            print(f"Strategy Alpha:        {(total_return - buy_hold_return) * 100:.2f}%")
        print(f"Number of Trades:      {len(self.trades)}")

        return {
            "portfolio_history": pd.DataFrame(self.portfolio_history),
            "trades": pd.DataFrame(self.trades),
            "final_value": final_value,
            "total_return": total_return,
            "buy_hold_return": buy_hold_return,
            "num_trades": len(self.trades),
        }


def predict_today(symbol, model_path="models", visualisation: bool = False):
    """
    Make predictions for today's trading using a saved model

    Args:
        symbol: Stock symbol to predict
        model_path: Path to saved models
        :param symbol:
        :param model_path:
        :param visualisation:
    """
    print("\n" + "=" * 80)
    print(f"LIVE PREDICTION FOR {symbol}")
    print("=" * 80)

    try:
        # Load the trained model
        print("\n1. Loading trained model...")
        predictor = XGBoostStockPredictor.load_model(symbol, model_path)

        # Download recent data (need enough history for indicators)
        print(f"\n2. Downloading recent data for {symbol}...")
        ticker = yf.Ticker(symbol)
        # Get last 100 days to calculate indicators
        df = ticker.history(period="100d", interval="1d")

        if df.empty:
            print(f"  ✗ No data available for {symbol}")
            return None

        print(f"  ✓ Downloaded {len(df)} days of data")
        print(f"  ✓ Latest date: {df.index[-1].date()}")
        print(f"  ✓ Latest close: ${df['Close'].iloc[-1]:.2f}")

        # Add technical indicators
        print("\n3. Calculating technical indicators...")
        df_features = FeatureEngineering.add_technical_indicators(df)

        # We don't need target for prediction
        feature_columns = [
            "SMA_5",
            "SMA_10",
            "SMA_20",
            "SMA_50",
            "EMA_12",
            "EMA_26",
            "MACD",
            "MACD_Signal",
            "MACD_Hist",
            "MACD_Momentum",
            "RSI",
            "Stochastic",
            "BB_Width",
            "Price_Change_1d",
            "Price_Change_5d",
            "Price_Change_10d",
            "Volume_Change",
            "Volume_Ratio",
            "Volume_SMA_Ratio",
            "Volatility_10d",
            "Volatility_20d",
            "Volatility_Ratio",
            "HL_Range",
            "ATR",
            "Return_Lag_1",
            "Return_Lag_2",
            "Return_Lag_3",
            "Return_Lag_5",
            "Return_Lag_10",
            "Momentum_5",
            "Momentum_10",
            "Momentum_20",
            "Price_to_SMA20",
            "Price_to_SMA50",
            "ROC_5",
            "ROC_10",
        ]

        # Get the latest complete row (has all indicators)
        df_clean = df_features.dropna()

        if len(df_clean) == 0:
            print("  ✗ Not enough data to calculate indicators")
            return None

        # Get features for the most recent day
        X_today = df_clean[feature_columns].iloc[[-1]]  # Keep as DataFrame
        latest_date = df_clean.index[-1]
        latest_price = df_clean["Close"].iloc[-1]

        print(f"  ✓ Using data from: {latest_date.date()}")

        # Make prediction
        print("\n4. Making prediction...")
        prediction = predictor.predict(X_today)[0]

        # Display results
        print("\n" + "=" * 80)
        print("PREDICTION RESULTS")
        print("=" * 80)
        print(f"\nSymbol:              {symbol}")
        print(f"Current Price:       ${latest_price:.2f}")
        print(f"Data Date:           {latest_date.date()}")
        print(f"\nPredicted Return:    {prediction * 100:+.3f}%")
        print(f"Predicted Price:     ${latest_price * (1 + prediction):.2f}")
        print(f"Expected Change:     ${latest_price * prediction:+.2f}")

        # Trading signal
        print("\n" + "-" * 80)
        print("TRADING SIGNAL")
        print("-" * 80)

        confidence = abs(prediction) / 0.02  # Normalized confidence (2% = high confidence)
        confidence = min(confidence, 1.0) * 100

        if prediction > 0.005:  # 0.5% threshold
            signal = "🟢 STRONG BUY"
            signal_text = "STRONG BUY"
            action = "Consider opening a LONG position"
        elif prediction > 0.002:
            signal = "🟢 BUY"
            signal_text = "BUY"
            action = "Consider opening a small LONG position"
        elif prediction > 0:
            signal = "🟡 WEAK BUY"
            signal_text = "WEAK BUY"
            action = "Slight bullish bias, monitor closely"
        elif prediction > -0.002:
            signal = "🟡 WEAK SELL"
            signal_text = "WEAK SELL"
            action = "Slight bearish bias, monitor closely"
        elif prediction > -0.005:
            signal = "🔴 SELL"
            signal_text = "SELL"
            action = "Consider closing LONG positions or small SHORT"
        else:
            signal = "🔴 STRONG SELL"
            signal_text = "STRONG SELL"
            action = "Consider closing LONG positions or opening SHORT"

        print(f"\nSignal:              {signal}")
        print(f"Confidence:          {confidence:.1f}%")
        print(f"Recommendation:      {action}")

        # Show key indicators
        print("\n" + "-" * 80)
        print("KEY INDICATORS (Latest)")
        print("-" * 80)
        print(f"RSI:                 {df_clean['RSI'].iloc[-1]:.2f}")
        print(f"MACD:                {df_clean['MACD'].iloc[-1]:.4f}")
        print(f"Price vs SMA20:      {df_clean['Price_to_SMA20'].iloc[-1]:.4f}x")
        print(f"Price vs SMA50:      {df_clean['Price_to_SMA50'].iloc[-1]:.4f}x")
        print(f"Volatility (10d):    {df_clean['Volatility_10d'].iloc[-1]:.2f}")

        print("\n" + "=" * 80)

        # Generate 180-day forecast
        print("\n5. Generating 180-day forecast...")
        forecast_days = 180
        forecast_data = generate_forecast(predictor, df_clean, feature_columns, latest_price, forecast_days)

        # Display forecast summary
        print(f"\n180-DAY FORECAST SUMMARY:")
        print(f"  Starting Price:      ${latest_price:.2f}")
        print(f"  Forecasted Price:    ${forecast_data['prices'][-1]:.2f}")
        print(f"  Expected Change:     ${forecast_data['prices'][-1] - latest_price:+.2f}")
        print(f"  Expected Return:     {(forecast_data['prices'][-1] / latest_price - 1) * 100:+.2f}%")
        print(
            f"  Confidence Range:    ${forecast_data['lower_bound'][-1]:.2f} - ${forecast_data['upper_bound'][-1]:.2f}"
        )

        # Generate forecast plot
        if visualisation:
            print("\n6. Creating forecast visualization...")
            Visualizer.plot_forecast(df, forecast_data, symbol)

        # Save prediction to file (use UTF-8 encoding and text-only signals)
        pred_file = f"results/{symbol}_latest_prediction.txt"
        with open(pred_file, "w", encoding="utf-8") as f:
            f.write(f"Latest Prediction for {symbol}\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data Date: {latest_date.date()}\n\n")
            f.write(f"Current Price:    ${latest_price:.2f}\n")
            f.write(f"Predicted Return: {prediction * 100:+.3f}%\n")
            f.write(f"Predicted Price:  ${latest_price * (1 + prediction):.2f}\n\n")
            f.write(f"Signal:           {signal_text}\n")
            f.write(f"Confidence:       {confidence:.1f}%\n")
            f.write(f"Recommendation:   {action}\n\n")

            # Add 180-day forecast
            f.write("=" * 60 + "\n")
            f.write("180-DAY FORECAST\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Starting Price:    ${latest_price:.2f}\n")
            f.write(f"180-Day Forecast:   ${forecast_data['prices'][-1]:.2f}\n")
            f.write(f"Expected Change:   ${forecast_data['prices'][-1] - latest_price:+.2f}\n")
            f.write(f"Expected Return:   {(forecast_data['prices'][-1] / latest_price - 1) * 100:+.2f}%\n")
            f.write(
                f"Confidence Range:  ${forecast_data['lower_bound'][-1]:.2f} - "
                f"${forecast_data['upper_bound'][-1]:.2f}\n\n"
            )

            f.write("Weekly Milestones:\n")
            for week in [0, 6, 13, 20, 29]:
                if week < len(forecast_data["prices"]):
                    f.write(
                        f"  Day {week + 1:2d}: ${forecast_data['prices'][week]:.2f} "
                        f"({(forecast_data['prices'][week] / latest_price - 1) * 100:+.2f}%)\n"
                    )

        print(f"✓ Prediction saved to: {pred_file}\n")

        return {
            "symbol": symbol,
            "date": latest_date,
            "current_price": latest_price,
            "predicted_return": prediction,
            "predicted_price": latest_price * (1 + prediction),
            "signal": signal,
            "confidence": confidence,
        }

    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print(f"  Please train a model for {symbol} first by running the main training script.")
        return None
    except Exception as e:
        print(f"\n✗ Error making prediction: {str(e)}")
        import traceback

        traceback.print_exc()
        return None


def generate_forecast(predictor, df_features, feature_columns, start_price, num_days=30):
    """
    Generate multi-day forecast by iteratively predicting and updating features

    Args:
        predictor: Trained XGBoostStockPredictor model
        df_features: DataFrame with all technical indicators
        feature_columns: List of feature column names
        start_price: Starting price for forecast
        num_days: Number of days to forecast

    Returns:
        Dictionary with forecast data
    """
    # Start with the most recent complete data
    current_features = df_features[feature_columns].iloc[-1:].copy()
    current_price = start_price

    forecast_prices = [current_price]
    forecast_returns = []
    forecast_dates = []

    # Get the last date and start forecasting from next day
    last_date = df_features.index[-1]

    # Calculate volatility for confidence intervals (use recent volatility)
    recent_returns = df_features["Price_Change_1d"].iloc[-30:].dropna()
    volatility = recent_returns.std()

    for day in range(num_days):
        # Make prediction for next day
        predicted_return = predictor.predict(current_features)[0]

        # Calculate next price
        next_price = current_price * (1 + predicted_return)

        # Store results
        next_date = last_date + timedelta(days=day + 1)
        forecast_dates.append(next_date)
        forecast_prices.append(next_price)
        forecast_returns.append(predicted_return)

        # Update features for next iteration
        # This is a simplified approach - in reality, we'd need to recalculate
        # all technical indicators with the new price point
        current_features = current_features.copy()

        # Update lagged returns
        if "Return_Lag_1" in current_features.columns:
            current_features["Return_Lag_10"] = current_features["Return_Lag_5"]
            current_features["Return_Lag_5"] = current_features["Return_Lag_3"]
            current_features["Return_Lag_3"] = current_features["Return_Lag_2"]
            current_features["Return_Lag_2"] = current_features["Return_Lag_1"]
            current_features["Return_Lag_1"] = predicted_return

        # Update price change features
        if "Price_Change_1d" in current_features.columns:
            current_features["Price_Change_1d"] = predicted_return

        # Add some momentum decay to prevent unrealistic trends
        decay_factor = 0.95  # Slight decay each day
        for col in current_features.columns:
            if "Momentum" in col or "ROC" in col:
                current_features[col] = current_features[col] * decay_factor

        current_price = next_price

    # Calculate confidence bounds (±1 standard deviation per day, compounding)
    forecast_prices_array = np.array(forecast_prices[1:])  # Exclude starting price
    days_forward = np.arange(1, num_days + 1)

    # Confidence grows with square root of time
    confidence_multiplier = volatility * np.sqrt(days_forward)
    lower_bound = forecast_prices_array * (1 - confidence_multiplier)
    upper_bound = forecast_prices_array * (1 + confidence_multiplier)

    return {
        "dates": forecast_dates,
        "prices": forecast_prices[1:],  # Exclude starting price
        "daily_returns": forecast_returns,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "volatility": volatility,
    }


def simulate_directional_trading_strategy(
    dates_test: XGBoostStockPredictor | Any,
    prices_test: np.ndarray[Any, np.dtype[Any]] | list[Any] | dict[str, float | Any] | Any,
    test_results: Series | Any,
    y_test: Series | Any,
) -> tuple[dict[str, DataFrame | float | int | Any], TradingSimulator]:
    # Strategy 1: Directional (most trades - buys on any positive prediction)
    print("\n--- Strategy 1: Directional Trading ---")
    print("Buys when prediction > 0, sells when prediction <= 0")
    simulator = TradingSimulator(initial_capital=10000)
    sim_results = simulator.simulate(
        test_results["predictions"],
        y_test,
        prices_test,
        dates_test,
        threshold=0,  # Not used in directional
        strategy="directional",
    )
    return sim_results, simulator


def simulate_adaptive_threshold_strategy(
    dates_test: XGBoostStockPredictor | Any,
    prices_test: np.ndarray[Any, np.dtype[Any]] | list[Any] | dict[str, float | Any] | Any,
    test_results: Series | Any,
    y_test: Series | Any,
) -> tuple[dict[str, DataFrame | float | int | Any], TradingSimulator]:
    # Strategy 2: Adaptive threshold
    print("\n--- Strategy 2: Adaptive Threshold ---")
    print("Uses statistical threshold based on prediction distribution")
    simulator = TradingSimulator(initial_capital=10000)
    adaptive_results = simulator.simulate(
        test_results["predictions"], y_test, prices_test, dates_test, threshold="adaptive", strategy="threshold"
    )
    return adaptive_results, simulator


def simulate_hold_days_strategy(
    dates_test: XGBoostStockPredictor | Any,
    prices_test: np.ndarray[Any, np.dtype[Any]] | list[Any] | dict[str, float | Any] | Any,
    test_results: Series | Any,
    y_test: Series | Any,
) -> tuple[dict[str, DataFrame | float | int | Any], TradingSimulator]:
    # Strategy 3: Hold days
    print("\n--- Strategy 3: Hold Days Strategy ---")
    print("Holds positions for multiple days")
    simulator = TradingSimulator(initial_capital=10000)
    hold_days_results = simulator.simulate(
        test_results["predictions"], y_test, prices_test, dates_test, threshold="adaptive", strategy="hold_days"
    )
    return hold_days_results, simulator
