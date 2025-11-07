import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings('ignore')

# Create directories for outputs
os.makedirs('historic_data', exist_ok=True)
os.makedirs('plots', exist_ok=True)
os.makedirs('results', exist_ok=True)


class StockDataDownloader:
    """Downloads and manages stock data from yfinance"""

    def __init__(self, symbols, period='2y', interval='1d'):
        self.symbols = symbols if isinstance(symbols, list) else [symbols]
        self.period = period
        self.interval = interval
        self.data = {}

    def download_data(self):
        """Download data for all symbols"""
        print("=" * 60)
        print("Downloading Stock Data from yfinance")
        print("=" * 60)

        for symbol in self.symbols:
            try:
                print(f"\nDownloading {symbol}...")
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=self.period, interval=self.interval)

                if df.empty:
                    print(f"  ⚠️  No data found for {symbol}")
                    continue

                # Save to CSV
                filename = f"historic_data/{symbol}_{self.period}_{self.interval}.csv"
                df.to_csv(filename)

                self.data[symbol] = df
                print(f"  ✓ Downloaded {len(df)} data points")
                print(f"  ✓ Date range: {df.index[0].date()} to {df.index[-1].date()}")
                print(f"  ✓ Saved to {filename}")

            except Exception as e:
                print(f"  ✗ Error downloading {symbol}: {str(e)}")

        return self.data


class FeatureEngineering:
    """Creates technical indicators and features for machine learning"""

    @staticmethod
    def add_technical_indicators(df):
        """Add various technical indicators to the dataframe"""
        df = df.copy()

        # Simple Moving Averages
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()

        # Exponential Moving Averages
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()

        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']

        # Price changes
        df['Price_Change_1d'] = df['Close'].pct_change(1)
        df['Price_Change_5d'] = df['Close'].pct_change(5)
        df['Price_Change_10d'] = df['Close'].pct_change(10)

        # Volume indicators
        df['Volume_Change'] = df['Volume'].pct_change(1)
        df['Volume_MA_5'] = df['Volume'].rolling(window=5).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_5']

        # Volatility
        df['Volatility_10d'] = df['Close'].rolling(window=10).std()
        df['Volatility_20d'] = df['Close'].rolling(window=20).std()

        # High-Low range
        df['HL_Range'] = (df['High'] - df['Low']) / df['Close']
        df['HL_Range_MA'] = df['HL_Range'].rolling(window=5).mean()

        # Average True Range (ATR)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(14).mean()

        # Additional predictive features
        # Lagged returns (previous days' returns as features)
        for lag in [1, 2, 3, 5, 10]:
            df[f'Return_Lag_{lag}'] = df['Close'].pct_change(1).shift(lag)

        # Price momentum indicators
        df['Momentum_5'] = df['Close'] - df['Close'].shift(5)
        df['Momentum_10'] = df['Close'] - df['Close'].shift(10)
        df['Momentum_20'] = df['Close'] - df['Close'].shift(20)

        # Price relative to moving averages
        df['Price_to_SMA20'] = df['Close'] / df['SMA_20']
        df['Price_to_SMA50'] = df['Close'] / df['SMA_50']

        # Volatility ratios
        df['Volatility_Ratio'] = df['Volatility_10d'] / df['Volatility_20d']

        # Volume trends
        df['Volume_SMA_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()

        # Rate of change
        df['ROC_5'] = ((df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5)) * 100
        df['ROC_10'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100

        # Stochastic Oscillator
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        df['Stochastic'] = 100 * (df['Close'] - low_14) / (high_14 - low_14)

        # MACD momentum
        df['MACD_Momentum'] = df['MACD'].diff()

        return df

    @staticmethod
    def create_target(df, target_days=1):
        """Create target variable (future returns)"""
        df = df.copy()
        df['Target'] = df['Close'].shift(-target_days) / df['Close'] - 1
        return df

    @staticmethod
    def prepare_features(df):
        """Prepare feature matrix and target vector"""
        feature_columns = [
            # Moving averages
            'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50',
            'EMA_12', 'EMA_26',
            # MACD indicators
            'MACD', 'MACD_Signal', 'MACD_Hist', 'MACD_Momentum',
            # Momentum and oscillators
            'RSI', 'Stochastic',
            # Bollinger Bands
            'BB_Width',
            # Price changes
            'Price_Change_1d', 'Price_Change_5d', 'Price_Change_10d',
            # Volume indicators
            'Volume_Change', 'Volume_Ratio', 'Volume_SMA_Ratio',
            # Volatility
            'Volatility_10d', 'Volatility_20d', 'Volatility_Ratio',
            # Other indicators
            'HL_Range', 'ATR',
            # Lagged returns
            'Return_Lag_1', 'Return_Lag_2', 'Return_Lag_3', 'Return_Lag_5', 'Return_Lag_10',
            # Momentum
            'Momentum_5', 'Momentum_10', 'Momentum_20',
            # Price ratios
            'Price_to_SMA20', 'Price_to_SMA50',
            # Rate of change
            'ROC_5', 'ROC_10'
        ]

        # Remove rows with NaN values
        df_clean = df.dropna()

        if len(df_clean) == 0:
            raise ValueError("No valid data after removing NaN values")

        X = df_clean[feature_columns]
        y = df_clean['Target']
        dates = df_clean.index
        prices = df_clean['Close']

        return X, y, dates, prices, feature_columns


class WalkForwardValidator:
    """Implements walk-forward analysis for time series"""

    def __init__(self, train_window=252, test_window=21, retrain_frequency=21):
        """
        Args:
            train_window: Number of days to use for training (252 = ~1 year)
            test_window: Number of days to test on before retraining
            retrain_frequency: How often to retrain (days)
        """
        self.train_window = train_window
        self.test_window = test_window
        self.retrain_frequency = retrain_frequency
        self.fold_results = []

    def split(self, X, y, dates):
        """Generate train/test splits for walk-forward"""
        splits = []
        n_samples = len(X)

        # Convert dates to list if it's an Index
        if hasattr(dates, 'tolist'):
            dates_list = dates.tolist()
        else:
            dates_list = list(dates)

        # Validate we have enough data
        min_required = self.train_window + self.test_window
        if n_samples < min_required:
            raise ValueError(
                f"Insufficient data for walk-forward analysis. "
                f"Need at least {min_required} samples "
                f"(train_window={self.train_window} + test_window={self.test_window}), "
                f"but only have {n_samples} samples."
            )

        # Start with initial training window
        start_idx = 0
        end_train_idx = self.train_window

        fold_num = 1
        while end_train_idx < n_samples:
            # Determine test period
            end_test_idx = min(end_train_idx + self.test_window, n_samples)

            if end_test_idx <= end_train_idx:
                break

            # Create split
            train_indices = list(range(start_idx, end_train_idx))
            test_indices = list(range(end_train_idx, end_test_idx))

            splits.append({
                'fold': fold_num,
                'train_indices': train_indices,
                'test_indices': test_indices,
                'train_dates': (dates_list[start_idx], dates_list[end_train_idx - 1]),
                'test_dates': (dates_list[end_train_idx], dates_list[end_test_idx - 1])
            })

            fold_num += 1

            # Move forward by retrain_frequency
            # Expanding window: keep start_idx at 0
            # Rolling window: start_idx += self.retrain_frequency
            end_train_idx += self.retrain_frequency

        if len(splits) == 0:
            raise ValueError(
                f"No valid folds generated. Data size: {n_samples}, "
                f"Train window: {self.train_window}, Test window: {self.test_window}"
            )

        return splits

    def validate(self, X, y, dates, prices, predictor_class, predictor_params=None):
        """
        Perform walk-forward validation

        Returns:
            Dictionary with combined predictions and fold information
        """
        print("\n" + "=" * 60)
        print("Walk-Forward Analysis")
        print("=" * 60)
        print(f"Train Window: {self.train_window} days (~{self.train_window/252:.1f} years)")
        print(f"Test Window: {self.test_window} days")
        print(f"Retrain Frequency: {self.retrain_frequency} days")

        splits = self.split(X, y, dates)
        print(f"Number of folds: {len(splits)}")

        all_predictions = []
        all_actuals = []
        all_dates = []
        all_prices = []
        fold_boundaries = []

        for split in splits:
            fold = split['fold']
            train_idx = split['train_indices']
            test_idx = split['test_indices']

            print(f"\nFold {fold}/{len(splits)}:")
            print(f"  Train: {split['train_dates'][0]} to {split['train_dates'][1]} ({len(train_idx)} samples)")
            print(f"  Test:  {split['test_dates'][0]} to {split['test_dates'][1]} ({len(test_idx)} samples)")

            # Get train/test data
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_test = y.iloc[test_idx]

            # Train model for this fold
            predictor = predictor_class(predictor_params)
            predictor.train(X_train, y_train)

            # Make predictions
            predictions = predictor.predict(X_test)

            # Calculate fold metrics
            fold_mse = mean_squared_error(y_test, predictions)
            fold_rmse = np.sqrt(fold_mse)
            fold_mae = mean_absolute_error(y_test, predictions)

            print(f"  Fold RMSE: {fold_rmse:.6f}")
            print(f"  Fold MAE:  {fold_mae:.6f}")

            # Store results
            all_predictions.extend(predictions)
            all_actuals.extend(y_test.values)

            # Handle dates indexing
            if hasattr(dates, 'tolist'):
                test_dates = [dates[i] for i in test_idx]
            else:
                test_dates = [dates[i] for i in test_idx]
            all_dates.extend(test_dates)
            all_prices.extend(prices.iloc[test_idx].values)

            fold_boundaries.append({
                'fold': fold,
                'start_date': split['test_dates'][0],
                'end_date': split['test_dates'][1],
                'retrain_date': split['train_dates'][1]
            })

            self.fold_results.append({
                'fold': fold,
                'rmse': fold_rmse,
                'mae': fold_mae,
                'n_train': len(train_idx),
                'n_test': len(test_idx)
            })

        # Calculate overall metrics
        overall_mse = mean_squared_error(all_actuals, all_predictions)
        overall_rmse = np.sqrt(overall_mse)
        overall_mae = mean_absolute_error(all_actuals, all_predictions)
        overall_r2 = r2_score(all_actuals, all_predictions)

        print(f"\n{'=' * 60}")
        print("Walk-Forward Overall Performance:")
        print(f"  RMSE: {overall_rmse:.8f}")
        print(f"  MAE:  {overall_mae:.8f}")
        print(f"  R²:   {overall_r2:.8f}")
        print(f"{'=' * 60}")

        return {
            'predictions': np.array(all_predictions),
            'actuals': np.array(all_actuals),
            'dates': all_dates,
            'prices': np.array(all_prices),
            'fold_boundaries': fold_boundaries,
            'metrics': {
                'rmse': overall_rmse,
                'mae': overall_mae,
                'r2': overall_r2,
                'mse': overall_mse
            }
        }


class XGBoostStockPredictor:
    """XGBoost model for stock price prediction"""

    def __init__(self, params=None):
        if params is None:
            params = {
                'objective': 'reg:squarederror',
                'max_depth': 50,
                'learning_rate': 0.05,
                'n_estimators': 1000,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'colsample_bylevel': 0.8,
                'gamma': 0.1,
                'reg_alpha': 0.5,
                'reg_lambda': 1.0,
                'min_child_weight': 3,
                'random_state': 42,
                'early_stopping_rounds': 50
            }
        self.params = params
        self.model = None
        self.feature_importance = None
        self.scaler = StandardScaler()

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the XGBoost model"""
        print("\n" + "=" * 60)
        print("Training XGBoost Model")
        print("=" * 60)

        # Scale features
        print("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)

        # Extract early stopping rounds if present
        early_stopping = self.params.pop('early_stopping_rounds', None)

        self.model = xgb.XGBRegressor(**self.params)

        if X_val is not None and y_val is not None and early_stopping is not None:
            X_val_scaled = self.scaler.transform(X_val)
            X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)

            eval_set = [(X_train_scaled, y_train), (X_val_scaled, y_val)]
            self.model.fit(
                X_train_scaled, y_train,
                eval_set=eval_set,
                verbose=False
            )
        else:
            self.model.fit(X_train_scaled, y_train)

        # Store feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        print(f"✓ Model trained successfully")
        print(f"✓ Number of trees: {self.model.n_estimators}")
        print(f"✓ Best iteration: {self.model.best_iteration if hasattr(self.model, 'best_iteration') else 'N/A'}")
        print(f"✓ Max depth: {self.params['max_depth']}")

    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet")

        # Scale features before prediction
        X_scaled = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

        return self.model.predict(X_scaled)

    def evaluate(self, X, y, dataset_name="Test"):
        """Evaluate model performance"""
        predictions = self.predict(X)

        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, predictions)
        r2 = r2_score(y, predictions)

        print(f"\n{dataset_name} Set Performance:")
        print(f"  MSE:  {mse:.8f}")
        print(f"  RMSE: {rmse:.8f}")
        print(f"  MAE:  {mae:.8f}")
        print(f"  R²:   {r2:.8f}")

        return {
            'predictions': predictions,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }


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


class Visualizer:
    """Creates visualizations for predictions and trading results"""

    @staticmethod
    def plot_predictions(dates, actual, predicted, symbol, save_path='plots'):
        """Plot actual vs predicted returns"""
        plt.figure(figsize=(15, 6))

        plt.subplot(1, 2, 1)
        plt.scatter(actual, predicted, alpha=0.5)
        plt.plot([actual.min(), actual.max()], 
                [actual.min(), actual.max()], 'r--', lw=2)
        plt.xlabel('Actual Returns')
        plt.ylabel('Predicted Returns')
        plt.title(f'{symbol} - Actual vs Predicted Returns')
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.plot(dates, actual, label='Actual', alpha=0.7)
        plt.plot(dates, predicted, label='Predicted', alpha=0.7)
        plt.xlabel('Date')
        plt.ylabel('Returns')
        plt.title(f'{symbol} - Returns Over Time')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{save_path}/{symbol}_predictions.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved predictions plot: {save_path}/{symbol}_predictions.png")
        plt.close()

    @staticmethod
    def plot_feature_importance(feature_importance, symbol, top_n=15, save_path='plots'):
        """Plot feature importance"""
        plt.figure(figsize=(10, 8))

        top_features = feature_importance.head(top_n)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.title(f'{symbol} - Top {top_n} Feature Importance')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        plt.savefig(f'{save_path}/{symbol}_feature_importance.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved feature importance plot: {save_path}/{symbol}_feature_importance.png")
        plt.close()

    @staticmethod
    def plot_trading_simulation(simulation_results, symbol, save_path='plots'):
        """Plot trading simulation results"""
        portfolio_df = simulation_results['portfolio_history']
        trades_df = simulation_results['trades']

        fig, axes = plt.subplots(3, 1, figsize=(15, 14))

        # Portfolio value over time
        ax1 = axes[0]
        ax1.plot(portfolio_df['date'], portfolio_df['portfolio_value'], 
                label='Portfolio Value', linewidth=2, color='blue')
        ax1.axhline(y=simulation_results['final_value'], 
                   color='g', linestyle='--', alpha=0.5, 
                   label=f"Final: ${simulation_results['final_value']:,.0f}")

        # Show buy & hold for comparison
        first_price = portfolio_df['price'].iloc[0]
        buy_hold_values = (portfolio_df['price'] / first_price) * simulation_results['final_value']
        ax1.plot(portfolio_df['date'], buy_hold_values, 
                label='Buy & Hold', linewidth=2, alpha=0.5, linestyle='--', color='orange')

        ax1.set_xlabel('Date')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.set_title(f'{symbol} - Portfolio Value Over Time (Trades: {len(trades_df)})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)

        # Stock price with buy/sell signals
        ax2 = axes[1]
        ax2.plot(portfolio_df['date'], portfolio_df['price'], 
                label='Stock Price', linewidth=2, alpha=0.7, color='black')

        if len(trades_df) > 0:
            buys = trades_df[trades_df['action'].str.contains('BUY', case=False, na=False)]
            sells = trades_df[trades_df['action'].str.contains('SELL', case=False, na=False)]

            if len(buys) > 0:
                ax2.scatter(buys['date'], buys['price'], 
                           color='green', marker='^', s=150, 
                           label=f'Buy ({len(buys)})', zorder=5, edgecolors='black', linewidth=1)
            if len(sells) > 0:
                ax2.scatter(sells['date'], sells['price'], 
                           color='red', marker='v', s=150, 
                           label=f'Sell ({len(sells)})', zorder=5, edgecolors='black', linewidth=1)
        else:
            # Add warning text if no trades
            ax2.text(0.5, 0.5, '⚠️ NO TRADES EXECUTED', 
                    transform=ax2.transAxes, ha='center', va='center',
                    fontsize=16, color='red', alpha=0.5, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

        ax2.set_xlabel('Date')
        ax2.set_ylabel('Stock Price ($)')
        ax2.set_title(f'{symbol} - Stock Price with Trading Signals')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)

        # Predictions over time
        ax3 = axes[2]
        if 'prediction' in portfolio_df.columns:
            predictions = portfolio_df['prediction'].values * 100  # Convert to percentage
            ax3.plot(portfolio_df['date'], predictions, 
                    label='Predicted Return (%)', linewidth=1.5, alpha=0.8, color='purple')
            ax3.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)

            # Fill areas
            ax3.fill_between(portfolio_df['date'], 0, predictions, 
                             where=(predictions > 0), alpha=0.3, color='green', label='Positive Prediction')
            ax3.fill_between(portfolio_df['date'], 0, predictions, 
                             where=(predictions < 0), alpha=0.3, color='red', label='Negative Prediction')

            # Add statistics
            pred_stats = f"Mean: {np.mean(predictions):.3f}% | Std: {np.std(predictions):.3f}% | Range: [{np.min(predictions):.3f}, {np.max(predictions):.3f}]%"
            ax3.text(0.02, 0.98, pred_stats, transform=ax3.transAxes, 
                    verticalalignment='top', fontsize=9, 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax3.set_xlabel('Date')
        ax3.set_ylabel('Predicted Return (%)')
        ax3.set_title(f'{symbol} - Model Predictions Over Time')
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(f'{save_path}/{symbol}_trading_simulation.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved trading simulation plot: {save_path}/{symbol}_trading_simulation.png")
        plt.close()

    @staticmethod
    def plot_walk_forward_results(wf_results, symbol, save_path='plots'):
        """Plot walk-forward analysis results with fold boundaries"""
        fig, axes = plt.subplots(3, 1, figsize=(16, 14))

        dates = wf_results['dates']
        predictions = wf_results['predictions']
        actuals = wf_results['actuals']
        prices = wf_results['prices']
        fold_boundaries = wf_results['fold_boundaries']

        # Plot 1: Predictions vs Actuals with fold boundaries
        ax1 = axes[0]
        ax1.plot(dates, actuals, label='Actual Returns', alpha=0.7, linewidth=1.5)
        ax1.plot(dates, predictions, label='Predicted Returns', alpha=0.7, linewidth=1.5)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)

        # Mark retrain points
        for boundary in fold_boundaries:
            ax1.axvline(x=boundary['start_date'], color='red', linestyle='--', 
                       alpha=0.3, linewidth=1)

        # Add legend entry for retrain markers
        ax1.axvline(x=dates[0], color='red', linestyle='--', 
                   alpha=0.5, linewidth=1, label='Retrain Point')

        ax1.set_xlabel('Date')
        ax1.set_ylabel('Returns')
        ax1.set_title(f'{symbol} - Walk-Forward: Predictions vs Actuals ({len(fold_boundaries)} folds)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)

        # Plot 2: Prediction errors over time
        ax2 = axes[1]
        errors = predictions - actuals
        ax2.plot(dates, errors, label='Prediction Error', alpha=0.7, linewidth=1, color='red')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax2.fill_between(dates, 0, errors, alpha=0.3, color='red')

        # Mark retrain points
        for boundary in fold_boundaries:
            ax2.axvline(x=boundary['start_date'], color='blue', linestyle='--', 
                       alpha=0.3, linewidth=1)

        # Add statistics
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors**2))
        stats_text = f'MAE: {mae:.6f} | RMSE: {rmse:.6f}'
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax2.set_xlabel('Date')
        ax2.set_ylabel('Error')
        ax2.set_title(f'{symbol} - Prediction Errors Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)

        # Plot 3: Cumulative returns comparison
        ax3 = axes[2]

        # Calculate cumulative returns
        actual_cumulative = (1 + pd.Series(actuals)).cumprod() - 1
        predicted_cumulative = (1 + pd.Series(predictions)).cumprod() - 1

        ax3.plot(dates, actual_cumulative * 100, label='Actual Cumulative Return', 
                linewidth=2, alpha=0.8)
        ax3.plot(dates, predicted_cumulative * 100, label='Predicted Cumulative Return', 
                linewidth=2, alpha=0.8)

        # Mark retrain points
        for boundary in fold_boundaries:
            ax3.axvline(x=boundary['start_date'], color='green', linestyle='--', 
                       alpha=0.3, linewidth=1)

        ax3.set_xlabel('Date')
        ax3.set_ylabel('Cumulative Return (%)')
        ax3.set_title(f'{symbol} - Cumulative Returns (Walk-Forward)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(f'{save_path}/{symbol}_walk_forward.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved walk-forward plot: {save_path}/{symbol}_walk_forward.png")
        plt.close()

    @staticmethod
    def plot_price_with_indicators(df, symbol, save_path='plots'):
        """Plot stock price with technical indicators"""
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))

        # Price with moving averages
        ax1 = axes[0]
        ax1.plot(df.index, df['Close'], label='Close', linewidth=2)
        ax1.plot(df.index, df['SMA_20'], label='SMA 20', alpha=0.7)
        ax1.plot(df.index, df['SMA_50'], label='SMA 50', alpha=0.7)
        ax1.fill_between(df.index, df['BB_Lower'], df['BB_Upper'], 
                         alpha=0.2, label='Bollinger Bands')
        ax1.set_ylabel('Price ($)')
        ax1.set_title(f'{symbol} - Price with Moving Averages')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # RSI
        ax2 = axes[1]
        ax2.plot(df.index, df['RSI'], linewidth=2)
        ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5)
        ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5)
        ax2.set_ylabel('RSI')
        ax2.set_title(f'{symbol} - Relative Strength Index')
        ax2.set_ylim([0, 100])
        ax2.grid(True, alpha=0.3)

        # Volume
        ax3 = axes[2]
        ax3.bar(df.index, df['Volume'], alpha=0.7)
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Volume')
        ax3.set_title(f'{symbol} - Trading Volume')
        ax3.grid(True, alpha=0.3)

        for ax in axes:
            ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(f'{save_path}/{symbol}_technical_indicators.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved technical indicators plot: {save_path}/{symbol}_technical_indicators.png")
        plt.close()


def main():
    """Main execution function"""
    print("\n" + "=" * 60)
    print("XGBoost Stock Price Prediction & Trading Simulation")
    print("=" * 60)

    # Configuration
    SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', "10M.L"]  # Stock symbols to analyze
    PERIOD = '1y'  # Data period
    INTERVAL = '1d'  # Data interval
    TEST_SIZE = 0.3  # Train/test split ratio
    PREDICTION_THRESHOLD = 0.002  # 0.2% threshold for trading signals

    # Walk-forward configuration
    USE_WALK_FORWARD = True  # Set to False for simple train/test split
    WALK_FORWARD_TRAIN_WINDOW = 180  # ~9 months of training data (reduced from 252)
    WALK_FORWARD_TEST_WINDOW = 20   # ~1 month of testing
    WALK_FORWARD_RETRAIN_FREQ = 20  # Retrain every ~1 month

    # Step 1: Download data
    downloader = StockDataDownloader(SYMBOLS, period=PERIOD, interval=INTERVAL)
    stock_data = downloader.download_data()

    if not stock_data:
        print("\n⚠️  No data downloaded. Exiting.")
        return

    # Process each symbol
    for symbol, df in stock_data.items():
        try:
            print("\n" + "=" * 60)
            print(f"Processing {symbol}")
            print("=" * 60)

            # Step 2: Feature engineering
            print("\nAdding technical indicators...")
            df_features = FeatureEngineering.add_technical_indicators(df)
            df_features = FeatureEngineering.create_target(df_features, target_days=1)

            # Step 3: Prepare features
            X, y, dates, prices, feature_columns = FeatureEngineering.prepare_features(df_features)
            print(f"✓ Prepared {len(X)} samples with {len(feature_columns)} features")

            # Step 4: Choose validation method
            use_walk_forward = USE_WALK_FORWARD

            # Check if we have enough data for walk-forward
            min_required = WALK_FORWARD_TRAIN_WINDOW + WALK_FORWARD_TEST_WINDOW
            if USE_WALK_FORWARD and len(X) < min_required:
                print(f"\n⚠️  Warning: Not enough data for walk-forward analysis")
                print(f"   Need {min_required} samples, have {len(X)} samples")
                print(f"   Falling back to simple train/test split")
                use_walk_forward = False

            if use_walk_forward:
                print("\n" + "=" * 60)
                print("Using Walk-Forward Analysis")
                print("=" * 60)

                try:
                    # Initialize walk-forward validator
                    wf_validator = WalkForwardValidator(
                        train_window=WALK_FORWARD_TRAIN_WINDOW,
                        test_window=WALK_FORWARD_TEST_WINDOW,
                        retrain_frequency=WALK_FORWARD_RETRAIN_FREQ
                    )

                    # Perform walk-forward validation
                    wf_results = wf_validator.validate(
                        X, y, dates, prices,
                        XGBoostStockPredictor,
                        predictor_params=None  # Uses default params
                    )
                except ValueError as e:
                    print(f"\n⚠️  Walk-forward analysis failed: {e}")
                    print(f"   Falling back to simple train/test split")
                    use_walk_forward = False

            if use_walk_forward:
                # Extract results for compatibility with existing code
                test_results = {
                    'predictions': wf_results['predictions'],
                    'rmse': wf_results['metrics']['rmse'],
                    'mae': wf_results['metrics']['mae'],
                    'r2': wf_results['metrics']['r2'],
                    'mse': wf_results['metrics']['mse']
                }

                y_test = pd.Series(wf_results['actuals'], index=wf_results['dates'])
                dates_test = wf_results['dates']
                prices_test = pd.Series(wf_results['prices'], index=wf_results['dates'])

                # Create a final model for feature importance (trained on all data)
                print("\nTraining final model on all data for feature importance...")
                predictor = XGBoostStockPredictor()
                split_idx = int(len(X) * 0.9)
                predictor.train(X.iloc[:split_idx], y.iloc[:split_idx], 
                              X.iloc[split_idx:], y.iloc[split_idx:])

                # Generate walk-forward specific plots
                print("\nGenerating walk-forward visualizations...")
                Visualizer.plot_walk_forward_results(wf_results, symbol)

            if not use_walk_forward:
                print("\n" + "=" * 60)
                print("Using Simple Train/Test Split")
                print("=" * 60)

                # Step 4: Train/test split
                split_idx = int(len(X) * (1 - TEST_SIZE))
                X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
                y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
                dates_test = dates[split_idx:]
                prices_test = prices[split_idx:]

                print(f"✓ Train set: {len(X_train)} samples")
                print(f"✓ Test set: {len(X_test)} samples")

                # Step 5: Train model
                predictor = XGBoostStockPredictor()
                predictor.train(X_train, y_train, X_test, y_test)

                # Step 6: Evaluate
                train_results = predictor.evaluate(X_train, y_train, "Train")
                test_results = predictor.evaluate(X_test, y_test, "Test")

            # Print final test results
            print(f"\nFinal Test Set Performance:")
            print(f"  RMSE: {test_results['rmse']:.8f}")
            print(f"  MAE:  {test_results['mae']:.8f}")
            print(f"  R²:   {test_results['r2']:.8f}")

            # Step 7: Generate plots
            print("\nGenerating visualizations...")
            Visualizer.plot_predictions(
                dates_test, y_test, test_results['predictions'], symbol
            )
            Visualizer.plot_feature_importance(
                predictor.feature_importance, symbol
            )
            Visualizer.plot_price_with_indicators(df_features, symbol)

            # Step 8: Trading simulation with multiple strategies
            print("\n" + "=" * 60)
            print("Testing Multiple Trading Strategies")
            print("=" * 60)

            # Strategy 1: Directional (most trades - buys on any positive prediction)
            print("\n--- Strategy 1: Directional Trading ---")
            print("Buys when prediction > 0, sells when prediction <= 0")
            simulator1 = TradingSimulator(initial_capital=10000)
            sim_results = simulator1.simulate(
                test_results['predictions'],
                y_test,
                prices_test,
                dates_test,
                threshold=0,  # Not used in directional
                strategy='directional'
            )

            # Strategy 2: Adaptive threshold
            print("\n--- Strategy 2: Adaptive Threshold ---")
            print("Uses statistical threshold based on prediction distribution")
            simulator2 = TradingSimulator(initial_capital=10000)
            sim_results2 = simulator2.simulate(
                test_results['predictions'],
                y_test,
                prices_test,
                dates_test,
                threshold='adaptive',
                strategy='threshold'
            )

            # Strategy 3: Hold days
            print("\n--- Strategy 3: Hold Days Strategy ---")
            print("Holds positions for multiple days")
            simulator3 = TradingSimulator(initial_capital=10000)
            sim_results3 = simulator3.simulate(
                test_results['predictions'],
                y_test,
                prices_test,
                dates_test,
                threshold='adaptive',
                strategy='hold_days'
            )

            # Compare strategies and use the best performing one for plots
            strategies = [
                ('Directional', sim_results, simulator1),
                ('Adaptive Threshold', sim_results2, simulator2),
                ('Hold Days', sim_results3, simulator3)
            ]

            # Find strategy with most trades (or best return if tied)
            valid_strategies = [(name, res, sim) for name, res, sim in strategies if res['num_trades'] > 0]

            if valid_strategies:
                best_strategy = max(valid_strategies, key=lambda x: (x[1]['num_trades'], x[1]['total_return']))
                print(f"\n{'=' * 60}")
                print(f"🏆 Best Strategy: {best_strategy[0]}")
                print(f"   Trades: {best_strategy[1]['num_trades']}")
                print(f"   Return: {best_strategy[1]['total_return']*100:.2f}%")
                print(f"{'=' * 60}")
                sim_results = best_strategy[1]
            else:
                print(f"\n⚠️  WARNING: No strategy generated trades!")
                sim_results = sim_results  # Use first strategy anyway for plotting

            # Step 9: Plot trading results
            Visualizer.plot_trading_simulation(sim_results, symbol)

            # Step 10: Save detailed results
            results_file = f'results/{symbol}_results.txt'
            with open(results_file, 'w') as f:
                f.write(f"XGBoost Stock Prediction Results for {symbol}\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Data Period: {PERIOD}\n")
                f.write(f"Total Samples: {len(X)}\n")
                f.write(f"Train Samples: {len(X_train)}\n")
                f.write(f"Test Samples: {len(X_test)}\n\n")
                f.write(f"Model Performance:\n")
                f.write(f"  Test RMSE: {test_results['rmse']:.8f}\n")
                f.write(f"  Test MAE: {test_results['mae']:.8f}\n")
                f.write(f"  Test R²: {test_results['r2']:.8f}\n\n")

                # Write results for all strategies
                f.write(f"Trading Simulation Results:\n")
                f.write(f"  Initial Capital: $10,000.00\n\n")

                for strategy_name, strategy_results, strategy_sim in strategies:
                    f.write(f"  {strategy_name} Strategy:\n")
                    f.write(f"    Final Value: ${strategy_results['final_value']:,.2f}\n")
                    f.write(f"    Total Return: {strategy_results['total_return']*100:.2f}%\n")
                    f.write(f"    Number of Trades: {strategy_results['num_trades']}\n")
                    if strategy_results['num_trades'] > 0:
                        alpha = (strategy_results['total_return'] - strategy_results['buy_hold_return']) * 100
                        f.write(f"    Alpha vs Buy&Hold: {alpha:.2f}%\n")
                    f.write("\n")

                f.write(f"  Buy & Hold Return: {sim_results['buy_hold_return']*100:.2f}%\n")

                if valid_strategies:
                    f.write(f"\n  Best Strategy: {best_strategy[0]}\n")
                    f.write(f"    Final Value: ${best_strategy[1]['final_value']:,.2f}\n")
                    f.write(f"    Total Return: {best_strategy[1]['total_return']*100:.2f}%\n")
                    f.write(f"    Number of Trades: {best_strategy[1]['num_trades']}\n")

            print(f"✓ Saved detailed results: {results_file}")

        except Exception as e:
            print(f"\n✗ Error processing {symbol}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "=" * 60)
    print("✓ All processing complete!")
    print(f"✓ Plots saved in: plots/")
    print(f"✓ Results saved in: results/")
    print(f"✓ CSV data saved in: historic_data/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
