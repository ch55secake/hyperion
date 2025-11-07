from datetime import datetime, timedelta

import yfinance as yf
import numpy as np
import pandas as pd

from src.feature import FeatureEngineering
from src.visualisation import Visualizer
from src.xbg.xgb_predictor import XGBoostStockPredictor


class TradingSimulator:
    """Simulates trading based on model predictions"""

    def __init__(self, initial_capital=10000, transaction_cost=0.001):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.portfolio_history = []
        self.trades = []

    def simulate(self, predictions, actual_returns, prices, dates, threshold='auto', strategy='directional'):
        """
        Simulate trading strategy

        Args:
            predictions: Predicted returns
            actual_returns: Actual returns
            prices: Stock prices
            dates: Dates
            threshold: Minimum predicted return to trigger a trade ('auto', 'adaptive', or float)
            strategy: 'directional' (trade on sign) or 'threshold' (trade above threshold)
        """
        print("\n" + "=" * 60)
        print("Running Trading Simulation")
        print("=" * 60)

        # Analyze predictions
        pred_array = np.array(predictions)
        print(f"\nPrediction Statistics:")
        print(f"  Mean:     {np.mean(pred_array):.6f} ({np.mean(pred_array)*100:.4f}%)")
        print(f"  Median:   {np.median(pred_array):.6f} ({np.median(pred_array)*100:.4f}%)")
        print(f"  Std Dev:  {np.std(pred_array):.6f} ({np.std(pred_array)*100:.4f}%)")
        print(f"  Min:      {np.min(pred_array):.6f} ({np.min(pred_array)*100:.4f}%)")
        print(f"  Max:      {np.max(pred_array):.6f} ({np.max(pred_array)*100:.4f}%)")
        print(f"  Range:    {np.max(pred_array) - np.min(pred_array):.6f}")

        # Count positive/negative predictions
        positive_preds = np.sum(pred_array > 0)
        negative_preds = np.sum(pred_array < 0)
        print(f"\nPrediction Distribution:")
        print(f"  Positive predictions: {positive_preds} ({positive_preds/len(pred_array)*100:.1f}%)")
        print(f"  Negative predictions: {negative_preds} ({negative_preds/len(pred_array)*100:.1f}%)")

        # Determine threshold
        if threshold == 'auto':
            # Use 25th percentile of absolute predictions
            threshold = np.percentile(np.abs(pred_array), 25)
            print(f"\nAuto threshold (25th percentile): {threshold:.6f} ({threshold*100:.4f}%)")
        elif threshold == 'adaptive':
            # Use 0.3 standard deviations
            threshold = 0.3 * np.std(pred_array)
            print(f"\nAdaptive threshold (0.3 std): {threshold:.6f} ({threshold*100:.4f}%)")
        else:
            threshold = float(threshold)
            print(f"\nFixed threshold: {threshold:.6f} ({threshold*100:.4f}%)")

        print(f"Strategy: {strategy}")

        # Count predictions above/below threshold
        above_threshold = np.sum(pred_array > threshold)
        below_threshold = np.sum(pred_array < -threshold)
        print(f"Predictions above +threshold: {above_threshold} ({above_threshold/len(pred_array)*100:.1f}%)")
        print(f"Predictions below -threshold: {below_threshold} ({below_threshold/len(pred_array)*100:.1f}%)")

        capital = self.initial_capital
        shares = 0
        position = None  # 'long' or None
        entry_price = 0
        hold_counter = 0

        for i in range(len(predictions)):
            date = dates.iloc[i] if hasattr(dates, 'iloc') else dates[i]
            pred_return = predictions[i]
            actual_return = actual_returns.iloc[i] if hasattr(actual_returns, 'iloc') else actual_returns[i]
            price = prices.iloc[i] if hasattr(prices, 'iloc') else prices[i]

            # Decision logic based on strategy
            if strategy == 'directional':
                # Simple strategy: buy if prediction is positive, sell if negative
                if position is None and pred_return > 0:
                    # Buy signal
                    shares = (capital * (1 - self.transaction_cost)) / price
                    entry_price = price
                    capital = 0
                    position = 'long'
                    self.trades.append({
                        'date': date,
                        'action': 'BUY',
                        'price': price,
                        'shares': shares,
                        'predicted_return': pred_return
                    })
                elif position == 'long' and pred_return <= 0:
                    # Sell signal
                    capital = shares * price * (1 - self.transaction_cost)
                    profit = capital - self.initial_capital
                    pnl = ((price - entry_price) / entry_price) * 100
                    self.trades.append({
                        'date': date,
                        'action': 'SELL',
                        'price': price,
                        'shares': shares,
                        'predicted_return': pred_return,
                        'profit': profit,
                        'pnl_pct': pnl
                    })
                    shares = 0
                    position = None

            elif strategy == 'threshold':
                # Threshold strategy: trade only when predictions exceed threshold
                if position is None:
                    if pred_return > threshold:
                        # Buy signal
                        shares = (capital * (1 - self.transaction_cost)) / price
                        entry_price = price
                        capital = 0
                        position = 'long'
                        self.trades.append({
                            'date': date,
                            'action': 'BUY',
                            'price': price,
                            'shares': shares,
                            'predicted_return': pred_return
                        })
                else:
                    # Exit if prediction reverses below negative threshold
                    if pred_return < -threshold:
                        capital = shares * price * (1 - self.transaction_cost)
                        profit = capital - self.initial_capital
                        pnl = ((price - entry_price) / entry_price) * 100
                        self.trades.append({
                            'date': date,
                            'action': 'SELL',
                            'price': price,
                            'shares': shares,
                            'predicted_return': pred_return,
                            'profit': profit,
                            'pnl_pct': pnl
                        })
                        shares = 0
                        position = None

            elif strategy == 'hold_days':
                # Hold for multiple days strategy
                if position is None and pred_return > threshold:
                    shares = (capital * (1 - self.transaction_cost)) / price
                    entry_price = price
                    capital = 0
                    position = 'long'
                    hold_counter = 0
                    self.trades.append({
                        'date': date,
                        'action': 'BUY',
                        'price': price,
                        'shares': shares,
                        'predicted_return': pred_return
                    })
                elif position == 'long':
                    hold_counter += 1
                    # Exit after 5 days or if prediction is strongly negative
                    if hold_counter >= 5 or pred_return < -threshold:
                        capital = shares * price * (1 - self.transaction_cost)
                        profit = capital - self.initial_capital
                        pnl = ((price - entry_price) / entry_price) * 100
                        self.trades.append({
                            'date': date,
                            'action': 'SELL',
                            'price': price,
                            'shares': shares,
                            'predicted_return': pred_return,
                            'profit': profit,
                            'pnl_pct': pnl,
                            'hold_days': hold_counter
                        })
                        shares = 0
                        position = None

            # Calculate portfolio value
            if position == 'long':
                portfolio_value = shares * price
            else:
                portfolio_value = capital

            self.portfolio_history.append({
                'date': date,
                'portfolio_value': portfolio_value,
                'position': position if position else 'cash',
                'price': price,
                'prediction': pred_return
            })

        # Close any remaining position at the end
        if position is not None:
            final_price = prices.iloc[-1] if hasattr(prices, 'iloc') else prices[-1]
            final_date = dates.iloc[-1] if hasattr(dates, 'iloc') else dates[-1]
            capital = shares * final_price * (1 - self.transaction_cost)
            profit = capital - self.initial_capital
            pnl = ((final_price - entry_price) / entry_price) * 100
            self.trades.append({
                'date': final_date,
                'action': 'SELL (Final)',
                'price': final_price,
                'shares': shares,
                'predicted_return': predictions[-1],
                'profit': profit,
                'pnl_pct': pnl
            })
            shares = 0
            position = None
            self.portfolio_history[-1]['portfolio_value'] = capital

        # Calculate performance metrics
        final_value = capital if capital > 0 else self.portfolio_history[-1]['portfolio_value']
        total_return = (final_value - self.initial_capital) / self.initial_capital

        # Buy and hold comparison
        first_price = prices.iloc[0] if hasattr(prices, 'iloc') else prices[0]
        last_price = prices.iloc[-1] if hasattr(prices, 'iloc') else prices[-1]
        buy_hold_return = (last_price - first_price) / first_price

        print(f"\n{'=' * 60}")
        print("Trading Simulation Results")
        print(f"{'=' * 60}")
        print(f"Initial Capital:       ${self.initial_capital:,.2f}")
        print(f"Final Portfolio Value: ${final_value:,.2f}")
        print(f"Total Return:          {total_return*100:.2f}%")
        print(f"Buy & Hold Return:     {buy_hold_return*100:.2f}%")
        print(f"Number of Trades:      {len(self.trades)}")

        if len(self.trades) > 0:
            print(f"Strategy Alpha:        {(total_return - buy_hold_return)*100:.2f}%")

            # Calculate win rate and average profit
            sell_trades = [t for t in self.trades if 'SELL' in t['action']]
            if len(sell_trades) > 0:
                profitable = sum(1 for t in sell_trades if t.get('profit', 0) > 0)
                win_rate = profitable / len(sell_trades) * 100
                avg_pnl = np.mean([t.get('pnl_pct', 0) for t in sell_trades])
                print(f"Win Rate:              {win_rate:.1f}% ({profitable}/{len(sell_trades)} trades)")
                print(f"Avg P&L per trade:     {avg_pnl:.2f}%")
        else:
            print("⚠️  WARNING: No trades executed!")
            print("    Possible reasons:")
            print(f"    - Predictions too small (max: {np.max(np.abs(pred_array))*100:.4f}%)")
            print(f"    - Threshold too high: {threshold*100:.4f}%")
            print("    - Try 'directional' strategy or lower threshold")

        return {
            'portfolio_history': pd.DataFrame(self.portfolio_history),
            'trades': pd.DataFrame(self.trades) if self.trades else pd.DataFrame(),
            'final_value': final_value,
            'total_return': total_return,
            'buy_hold_return': buy_hold_return,
            'num_trades': len(self.trades)
        }


def predict_today(symbol, model_path='models'):
    """
    Make predictions for today's trading using a saved model

    Args:
        symbol: Stock symbol to predict
        model_path: Path to saved models
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
        df = ticker.history(period='100d', interval='1d')

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
            'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50',
            'EMA_12', 'EMA_26',
            'MACD', 'MACD_Signal', 'MACD_Hist', 'MACD_Momentum',
            'RSI', 'Stochastic',
            'BB_Width',
            'Price_Change_1d', 'Price_Change_5d', 'Price_Change_10d',
            'Volume_Change', 'Volume_Ratio', 'Volume_SMA_Ratio',
            'Volatility_10d', 'Volatility_20d', 'Volatility_Ratio',
            'HL_Range', 'ATR',
            'Return_Lag_1', 'Return_Lag_2', 'Return_Lag_3', 'Return_Lag_5', 'Return_Lag_10',
            'Momentum_5', 'Momentum_10', 'Momentum_20',
            'Price_to_SMA20', 'Price_to_SMA50',
            'ROC_5', 'ROC_10'
        ]

        # Get the latest complete row (has all indicators)
        df_clean = df_features.dropna()

        if len(df_clean) == 0:
            print("  ✗ Not enough data to calculate indicators")
            return None

        # Get features for the most recent day
        X_today = df_clean[feature_columns].iloc[[-1]]  # Keep as DataFrame
        latest_date = df_clean.index[-1]
        latest_price = df_clean['Close'].iloc[-1]

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
        print(f"\nPredicted Return:    {prediction*100:+.3f}%")
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

        # Generate 30-day forecast
        print("\n5. Generating 30-day forecast...")
        forecast_days = 180
        forecast_data = generate_forecast(predictor, df_clean, feature_columns,
                                          latest_price, forecast_days)

        # Display forecast summary
        print(f"\n30-DAY FORECAST SUMMARY:")
        print(f"  Starting Price:      ${latest_price:.2f}")
        print(f"  Forecasted Price:    ${forecast_data['prices'][-1]:.2f}")
        print(f"  Expected Change:     ${forecast_data['prices'][-1] - latest_price:+.2f}")
        print(f"  Expected Return:     {(forecast_data['prices'][-1]/latest_price - 1)*100:+.2f}%")
        print(f"  Confidence Range:    ${forecast_data['lower_bound'][-1]:.2f} - ${forecast_data['upper_bound'][-1]:.2f}")

        # Generate forecast plot
        print("\n6. Creating forecast visualization...")
        Visualizer.plot_forecast(df, forecast_data, symbol)

        # Save prediction to file (use UTF-8 encoding and text-only signals)
        pred_file = f'results/{symbol}_latest_prediction.txt'
        with open(pred_file, 'w', encoding='utf-8') as f:
            f.write(f"Latest Prediction for {symbol}\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data Date: {latest_date.date()}\n\n")
            f.write(f"Current Price:    ${latest_price:.2f}\n")
            f.write(f"Predicted Return: {prediction*100:+.3f}%\n")
            f.write(f"Predicted Price:  ${latest_price * (1 + prediction):.2f}\n\n")
            f.write(f"Signal:           {signal_text}\n")
            f.write(f"Confidence:       {confidence:.1f}%\n")
            f.write(f"Recommendation:   {action}\n\n")

            # Add 30-day forecast
            f.write("=" * 60 + "\n")
            f.write("30-DAY FORECAST\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Starting Price:    ${latest_price:.2f}\n")
            f.write(f"30-Day Forecast:   ${forecast_data['prices'][-1]:.2f}\n")
            f.write(f"Expected Change:   ${forecast_data['prices'][-1] - latest_price:+.2f}\n")
            f.write(f"Expected Return:   {(forecast_data['prices'][-1]/latest_price - 1)*100:+.2f}%\n")
            f.write(f"Confidence Range:  ${forecast_data['lower_bound'][-1]:.2f} - ${forecast_data['upper_bound'][-1]:.2f}\n\n")

            f.write("Weekly Milestones:\n")
            for week in [0, 6, 13, 20, 29]:
                if week < len(forecast_data['prices']):
                    f.write(f"  Day {week+1:2d}: ${forecast_data['prices'][week]:.2f} "
                            f"({(forecast_data['prices'][week]/latest_price - 1)*100:+.2f}%)\n")

        print(f"✓ Prediction saved to: {pred_file}\n")

        return {
            'symbol': symbol,
            'date': latest_date,
            'current_price': latest_price,
            'predicted_return': prediction,
            'predicted_price': latest_price * (1 + prediction),
            'signal': signal,
            'confidence': confidence
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
    recent_returns = df_features['Price_Change_1d'].iloc[-30:].dropna()
    volatility = recent_returns.std()

    for day in range(num_days):
        # Make prediction for next day
        predicted_return = predictor.predict(current_features)[0]

        # Calculate next price
        next_price = current_price * (1 + predicted_return)

        # Store results
        next_date = last_date + timedelta(days=day+1)
        forecast_dates.append(next_date)
        forecast_prices.append(next_price)
        forecast_returns.append(predicted_return)

        # Update features for next iteration
        # This is a simplified approach - in reality, we'd need to recalculate
        # all technical indicators with the new price point
        current_features = current_features.copy()

        # Update lagged returns
        if 'Return_Lag_1' in current_features.columns:
            current_features['Return_Lag_10'] = current_features['Return_Lag_5']
            current_features['Return_Lag_5'] = current_features['Return_Lag_3']
            current_features['Return_Lag_3'] = current_features['Return_Lag_2']
            current_features['Return_Lag_2'] = current_features['Return_Lag_1']
            current_features['Return_Lag_1'] = predicted_return

        # Update price change features
        if 'Price_Change_1d' in current_features.columns:
            current_features['Price_Change_1d'] = predicted_return

        # Add some momentum decay to prevent unrealistic trends
        decay_factor = 0.95  # Slight decay each day
        for col in current_features.columns:
            if 'Momentum' in col or 'ROC' in col:
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
        'dates': forecast_dates,
        'prices': forecast_prices[1:],  # Exclude starting price
        'daily_returns': forecast_returns,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'volatility': volatility
    }

