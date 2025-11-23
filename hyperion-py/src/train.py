import json
import os
import traceback
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

from src.data import StockDataDownloader
from src.feature import FeatureEngineering
from src.lgb import LightGBMStockPredictor
from src.optimise import StockModelOptimizer
from src.stacker import StackedStockPredictor
from src.visualisation import generate_plots, Visualizer
from src.writer import save_trained_model, persist_results, output_best_strategy
from src.xbg import XGBoostStockPredictor

TEST_SIZE = 0.3  # Train/test split ratio
USE_WALK_FORWARD = False  # Set to False for a simple train / test split


def load_best_params_file(symbol: str, model_type: str):
    with open(f"params/{symbol}_best_params.json", "r") as f:
        params = json.load(f)
        return params[model_type]["best_params"]


def simple_train_test_split(
    x_daily: pd.DataFrame, x_hourly: pd.DataFrame, dates: pd.Series, prices: pd.Series, y: pd.Series, symbol: str = ""
) -> Tuple[
    Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], pd.Series, StackedStockPredictor, pd.Series, dict, pd.Series
]:

    split_idx = int(len(x_daily) * (1 - TEST_SIZE))

    x_train_daily, x_test_daily = x_daily.iloc[:split_idx], x_daily.iloc[split_idx:]
    x_train_hourly, x_test_hourly = x_hourly.iloc[:split_idx], x_hourly.iloc[split_idx:]

    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    dates_test = dates[split_idx:]
    prices_test = prices.iloc[split_idx:]

    optimizer = StockModelOptimizer(x_train_daily, y_train, x_test_daily, y_test)

    optimizer.optimize_both()

    # Visualize
    optimizer.visualize_studies(save_path="plots/optuna")

    # Save results
    optimizer.save_results(f"params/{symbol}_best_params.json")

    x_train_dict = {"daily": x_train_daily, "hourly": x_train_hourly}
    x_test_dict = {"daily": x_test_daily, "hourly": x_test_hourly}

    # Create stacked predictor
    stacked = StackedStockPredictor(
        {
            "daily": XGBoostStockPredictor(),
            "hourly": LightGBMStockPredictor(),
        }
    )

    # Train base models
    train_data = {
        "daily": (x_train_daily, y_train, x_test_daily, y_test),
        "hourly": (x_train_hourly, y_train, x_test_hourly, y_test),
    }
    stacked.train(train_data)

    # Evaluate on test set
    test_results = stacked.evaluate(x_test_dict, y_test)

    return x_test_dict, x_train_dict, dates_test, stacked, prices_test, test_results, y_test


def train_model(symbols=None, period: str = "5y", interval: str = "1h", visualization: bool = False):
    """
    Train stacked model for each ticker, using daily + hourly features.
    """
    if symbols is None:
        symbols: list[str] = []
        with open("resources/tickers.txt", "r") as f:
            for line in f:
                symbols.append(line.strip())
        # symbols = ["AAPL"]
    print("\n" + "=" * 60)
    print("Stacked Stock Price Prediction (Daily + Hourly)")
    print("=" * 60)

    # Download data hourly (interval='1h')
    stock_data_downloader = StockDataDownloader(symbols, period="2y", interval=interval)

    stock_data_hourly = stock_data_downloader.download_data()

    if not stock_data_hourly:
        print("⚠️  No data downloaded. Exiting.")
        return False

    for symbol in symbols:
        if visualization:
            os.makedirs("./plots/" + symbol, exist_ok=True)
        try:
            print("\n" + "=" * 60)
            print(f"Processing {symbol}")
            print("=" * 60)

            # Sector, market cap, industry and avg_volume

            # Daily features
            features_daily = FeatureEngineering(stock_data_hourly[symbol])
            df_daily = features_daily.create_target_features()
            x_daily, y_daily, dates_daily, prices_daily, _ = features_daily.prepare_features()

            # Sector, market cap, industry and avg_volume
            # TODO?
            # x_daily["ticker"] = symbol
            # x_daily["ticker"] = x_daily["ticker"].astype("category")
            x_daily["sector"] = stock_data_downloader.get_sector(symbol)
            x_daily["sector"] = x_daily["sector"].astype("category")
            x_daily["industry"] = stock_data_downloader.get_industry(symbol)
            x_daily["industry"] = x_daily["industry"].astype("category")
            x_daily["beta"] = stock_data_downloader.get_beta(symbol)
            x_daily["avg_volume_log"] = np.log(stock_data_downloader.get_avg_volume(symbol))
            x_daily["market_cap"] = np.log(stock_data_downloader.get_market_cap(symbol))

            print(x_daily.head())

            # Hourly features
            features_hourly = FeatureEngineering(stock_data_hourly[symbol])
            features_hourly.create_target_features()
            x_hourly, _, _, _, _ = features_hourly.prepare_features()

            # Sector, market cap, industry and avg_volume
            # TODO?
            # x_hourly["ticker"] = symbol
            # x_hourly["ticker"] = x_hourly["ticker"].astype("category")
            x_hourly["sector"] = stock_data_downloader.get_sector(symbol)
            x_hourly["sector"] = x_hourly["sector"].astype("category")
            x_hourly["industry"] = stock_data_downloader.get_industry(symbol)
            x_hourly["industry"] = x_hourly["industry"].astype("category")
            x_hourly["beta"] = stock_data_downloader.get_beta(symbol)
            x_hourly["avg_volume_log"] = np.log(stock_data_downloader.get_avg_volume(symbol))
            x_hourly["market_cap"] = np.log(stock_data_downloader.get_market_cap(symbol))

            # Align hourly target with daily (optional: forward-fill or aggregate)
            # Here we just slice to daily index for stacking
            x_hourly = x_hourly.loc[x_daily.index]

            x_test_dict, x_train_dict, dates_test, predictor, prices_test, test_results, y_test = (
                simple_train_test_split(x_daily, x_hourly, dates_daily, prices_daily, y_daily, symbol)
            )

            save_trained_model(predictor, symbol, test_results)

            if visualization:
                generate_plots(dates_test, df_daily, predictor, symbol, test_results, y_test)

            run_trade_simulation(
                dates_test,
                period,
                prices_test,
                symbol,
                test_results,
                visualization,
                x_daily,
                x_test_dict,
                x_train_dict,
                y_test,
            )

        except Exception as e:
            print(f"\n✗ Error processing {symbol}: {str(e)}")
            traceback.print_exc()
            return False

    print("\n" + "=" * 60)
    print("✓ All processing complete!")
    return True


def run_trade_simulation(
    dates_test,
    period: str,
    prices_test,
    symbol: str,
    test_results: dict,
    visualization: bool,
    x,
    x_test_dict: dict,
    x_train_dict: dict,
    y_test,
):
    print("\n" + "=" * 60)
    print("Testing Multiple Trading Strategies")
    print("=" * 60)

    preds = test_results.get("predictions")
    if preds is None:
        print("⚠️  Predictions missing, computing via predictor.predict()")
        predictor = x_test_dict.get("predictor")
        preds = predictor.predict(x_test_dict)

    # Ensure 1D
    preds = np.asarray(preds).ravel()
    test_results["predictions"] = preds

    from src.simulation import TradingSimulator
    from src.simulation.strategy.directional import DirectionalTradingStrategy
    from src.simulation.strategy.adaptive import AdaptiveThresholdStrategy
    from src.simulation.strategy.hold_days import HoldDaysStrategy
    from src.simulation.strategy.strategy import Strategy

    # Initialize capital for all strategies
    initial_capital = 10000

    # Run directional strategy
    print("\n--- Strategy 1: Directional Trading ---")
    print("Buys when prediction > 0, sells when prediction <= 0")
    directional_simulator = TradingSimulator(initial_capital=initial_capital)
    directional_strategy = DirectionalTradingStrategy(directional_simulator, initial_capital)
    directional_trading_results, directional_simulator = Strategy.simulate(
        directional_strategy, dates_test, prices_test, preds, y_test
    )

    # Run adaptive threshold strategy
    print("\n--- Strategy 2: Adaptive Threshold ---")
    print("Uses statistical threshold based on prediction distribution")
    adaptive_simulator = TradingSimulator(initial_capital=initial_capital)
    adaptive_strategy = AdaptiveThresholdStrategy(
        adaptive_simulator, initial_capital, threshold=0.3 * np.std(np.array(preds))
    )
    adaptive_threshold_results, adaptive_simulator = Strategy.simulate(
        adaptive_strategy, dates_test, prices_test, preds, y_test
    )

    # Run hold days strategy
    print("\n--- Strategy 3: Hold Days Strategy ---")
    print("Holds positions for multiple days")
    hold_days_simulator = TradingSimulator(initial_capital=initial_capital)
    hold_days_strategy = HoldDaysStrategy(
        hold_days_simulator, initial_capital, hold_days=5, threshold=0.3 * np.std(np.array(preds))
    )
    hold_days_results, hold_days_simulator = Strategy.simulate(
        hold_days_strategy, dates_test, prices_test, preds, y_test
    )

    # Compare strategies and persist
    strategies = [
        ("Directional", directional_trading_results, directional_simulator),
        ("Adaptive Threshold", adaptive_threshold_results, adaptive_simulator),
        ("Hold Days", hold_days_results, hold_days_simulator),
    ]

    valid_strategies = [(name, res, sim) for name, res, sim in strategies if res["num_trades"] > 0]

    if valid_strategies:
        best_strategy, sim_results = output_best_strategy(valid_strategies)
        if best_strategy:
            persist_results(
                x,
                # TODO: This is why the number of samples are coming out wrong when we save the results
                x_test_dict,
                x_train_dict,
                best_strategy,
                period,
                sim_results,
                strategies,
                symbol,
                test_results,
                valid_strategies,
            )
    else:
        print("⚠️  No strategy generated trades, using directional as fallback")
        sim_results = directional_trading_results

    if visualization:
        Visualizer.plot_trading_simulation(sim_results, symbol)

        df_features_numeric = x.select_dtypes(include=[np.number])
        if not df_features_numeric.empty:
            Visualizer.plot_correlation_heatmap(df_features_numeric, symbol)

        Visualizer.plot_rolling_portfolio_metrics(sim_results["portfolio_history"], symbol)
        Visualizer.plot_drawdowns(sim_results["portfolio_history"], symbol)
        Visualizer.plot_win_loss_over_time(sim_results["trades"], symbol)


def prepare_all_stocks_data(symbols: List[str], period: str = "2y", interval: str = "1h", test_size: float = 0.2):
    """
    Download and prepare data for all stocks, splitting each stock individually by time

    Returns:
        Dictionary with train and test data for all stocks
    """
    print("\n" + "=" * 60)
    print("Downloading and Preparing Data for All Stocks")
    print("=" * 60)

    stock_data_downloader = StockDataDownloader(symbols, period=period, interval=interval)
    stock_data = stock_data_downloader.download_data()

    if not stock_data:
        print("⚠️  No data downloaded. Exiting.")
        return None

    # Separate lists for train and test data
    train_daily_features = []
    train_hourly_features = []
    train_targets = []
    train_dates = []
    train_prices = []
    train_symbols = []

    test_daily_features = []
    test_hourly_features = []
    test_targets = []
    test_dates = []
    test_prices = []
    test_symbols = []

    for symbol in symbols:
        try:
            print(f"\nProcessing {symbol}...")

            # Daily features
            features_daily = FeatureEngineering(stock_data[symbol])
            features_daily.create_target_features()
            x_daily, y_daily, dates_daily, prices_daily, _ = features_daily.prepare_features()

            # Add stock-specific features
            x_daily["ticker"] = symbol
            x_daily["sector"] = stock_data_downloader.get_sector(symbol)
            x_daily["industry"] = stock_data_downloader.get_industry(symbol)
            x_daily["beta"] = stock_data_downloader.get_beta(symbol)
            x_daily["avg_volume_log"] = np.log(stock_data_downloader.get_avg_volume(symbol) + 1)
            x_daily["market_cap_log"] = np.log(stock_data_downloader.get_market_cap(symbol) + 1)

            # Hourly features
            features_hourly = FeatureEngineering(stock_data[symbol])
            features_hourly.create_target_features()
            x_hourly, _, _, _, _ = features_hourly.prepare_features()

            # Add stock-specific features to hourly
            x_hourly["ticker"] = symbol
            x_hourly["sector"] = stock_data_downloader.get_sector(symbol)
            x_hourly["industry"] = stock_data_downloader.get_industry(symbol)
            x_hourly["beta"] = stock_data_downloader.get_beta(symbol)
            x_hourly["avg_volume_log"] = np.log(stock_data_downloader.get_avg_volume(symbol) + 1)
            x_hourly["market_cap_log"] = np.log(stock_data_downloader.get_market_cap(symbol) + 1)

            # Align hourly with daily
            x_hourly = x_hourly.loc[x_daily.index]

            # Split THIS stock's data by time
            split_idx = int(len(x_daily) * (1 - test_size))

            # Train data for this stock
            train_daily_features.append(x_daily.iloc[:split_idx])
            train_hourly_features.append(x_hourly.iloc[:split_idx])
            train_targets.append(y_daily.iloc[:split_idx])
            train_dates.append(pd.Series(dates_daily[:split_idx], index=dates_daily[:split_idx]))
            train_prices.append(prices_daily.iloc[:split_idx])
            train_symbols.extend([symbol] * split_idx)

            # Test data for this stock
            test_daily_features.append(x_daily.iloc[split_idx:])
            test_hourly_features.append(x_hourly.iloc[split_idx:])
            test_targets.append(y_daily.iloc[split_idx:])
            test_dates.append(pd.Series(dates_daily[split_idx:], index=dates_daily[split_idx:]))
            test_prices.append(prices_daily.iloc[split_idx:])
            test_symbols.extend([symbol] * (len(x_daily) - split_idx))

            print(f"  ✓ {symbol}: {split_idx} train samples, {len(x_daily) - split_idx} test samples")

        except Exception as e:
            print(f"  ✗ Error processing {symbol}: {str(e)}")
            traceback.print_exc()
            continue

    # Combine train data
    print("\n" + "=" * 60)
    print("Combining Training Data")
    print("=" * 60)

    train_daily = pd.concat(train_daily_features, axis=0, ignore_index=False)
    train_hourly = pd.concat(train_hourly_features, axis=0, ignore_index=False)
    train_targets = pd.concat(train_targets, axis=0, ignore_index=False)
    train_dates = pd.concat(train_dates, axis=0, ignore_index=False)
    train_prices = pd.concat(train_prices, axis=0, ignore_index=False)
    train_symbols_series = pd.Series(train_symbols, index=train_daily.index)

    # Combine test data
    print("Combining Test Data")

    test_daily = pd.concat(test_daily_features, axis=0, ignore_index=False)
    test_hourly = pd.concat(test_hourly_features, axis=0, ignore_index=False)
    test_targets = pd.concat(test_targets, axis=0, ignore_index=False)
    test_dates = pd.concat(test_dates, axis=0, ignore_index=False)
    test_prices = pd.concat(test_prices, axis=0, ignore_index=False)
    test_symbols_series = pd.Series(test_symbols, index=test_daily.index)

    # Convert categorical columns to category dtype AFTER concatenation
    print("\nConverting categorical columns...")
    categorical_cols = ["ticker", "sector", "industry"]
    for col in categorical_cols:
        if col in train_daily.columns:
            train_daily[col] = train_daily[col].astype("category")
            test_daily[col] = test_daily[col].astype("category")
            print(f"  {col}: {train_daily[col].nunique()} unique values")
        if col in train_hourly.columns:
            train_hourly[col] = train_hourly[col].astype("category")
            test_hourly[col] = test_hourly[col].astype("category")

    print(f"✓ Total train samples: {len(train_daily)}")
    print(f"✓ Total test samples: {len(test_daily)}")
    print(f"✓ Number of stocks: {len(symbols)}")
    print(f"✓ Stocks in test set: {test_symbols_series.nunique()}")
    print(f"✓ Features per timeframe: {len(train_daily.columns)}")

    return {
        "train": {
            "daily": train_daily,
            "hourly": train_hourly,
            "targets": train_targets,
            "dates": train_dates,
            "prices": train_prices,
            "symbols": train_symbols_series,
        },
        "test": {
            "daily": test_daily,
            "hourly": test_hourly,
            "targets": test_targets,
            "dates": test_dates,
            "prices": test_prices,
            "symbols": test_symbols_series,
        },
    }


def train_single_model_for_all_stocks(
    symbols=None, period: str = "2y", interval: str = "1h", test_size: float = 0.2, visualization: bool = False
):
    """
    Train a SINGLE stacked model for ALL tickers using combined data
    """
    if symbols is None:
        symbols = []
        with open("resources/tickers.txt", "r") as f:
            for line in f:
                symbols.append(line.strip())

    print("\n" + "=" * 60)
    print("Single Model Training for All Stocks")
    print("=" * 60)

    # Prepare combined data for all stocks (now returns train/test split)
    split_data = prepare_all_stocks_data(symbols, period, interval, test_size)

    if split_data is None:
        return False

    # Extract train and test data
    x_train_daily = split_data["train"]["daily"]
    x_train_hourly = split_data["train"]["hourly"]
    y_train = split_data["train"]["targets"]

    x_test_daily = split_data["test"]["daily"]
    x_test_hourly = split_data["test"]["hourly"]
    y_test = split_data["test"]["targets"]
    dates_test = split_data["test"]["dates"]
    prices_test = split_data["test"]["prices"]
    symbols_test = split_data["test"]["symbols"]

    print("\n" + "=" * 60)
    print("Training Single Model")
    print("=" * 60)
    print(f"Training samples: {len(x_train_daily)}")
    print(f"Testing samples: {len(x_test_daily)}")
    print(f"Unique stocks in test set: {symbols_test.nunique()}")

    # Optional: Hyperparameter optimization (disabled by default for single model)
    xgb_params = None
    lgb_params = None

    # Enable optimization here, but note it may take a long time
    use_optuna = False

    if use_optuna:
        optimizer = StockModelOptimizer(x_train_daily, y_train, x_test_daily, y_test, n_trials=1000, n_jobs=1)
        optimizer.optimize_both()
        optimizer.visualize_studies(save_path="plots/optuna")
        optimizer.save_results(f"params/ALL_STOCKS_best_params.json")

        xgb_params, lgb_params = optimizer.best_xgb_params, optimizer.best_lgb_params

    # Create stacked predictor with default or optimized params
    stacked = StackedStockPredictor(
        {
            "daily": XGBoostStockPredictor(params=xgb_params),
            "hourly": LightGBMStockPredictor(params=lgb_params),
        }
    )

    # Train base models on ALL stock data
    train_data = {
        "daily": (x_train_daily, y_train, x_test_daily, y_test),
        "hourly": (x_train_hourly, y_train, x_test_hourly, y_test),
    }
    stacked.train(train_data)

    # Evaluate on test set
    x_test_dict = {"daily": x_test_daily, "hourly": x_test_hourly}
    test_results = stacked.evaluate(x_test_dict, y_test)

    # Save the single model (use a generic name like "ALL_STOCKS")
    model_name = "ALL_STOCKS"
    save_trained_model(stacked, model_name, test_results)

    print("\n" + "=" * 60)
    print("✓ Single model training complete!")
    print(f"✓ Model saved as: {model_name}")
    print("=" * 60)

    # Optional: Evaluate per-stock performance
    if visualization:
        evaluate_per_stock_performance(stacked, x_test_dict, y_test, symbols_test, dates_test, prices_test)

    return True


def evaluate_per_stock_performance(model, x_test_dict, y_test, symbols_test, dates_test, prices_test):
    """
    Evaluate model performance for each stock individually
    """
    print("\n" + "=" * 60)
    print("Per-Stock Performance Analysis")
    print("=" * 60)

    predictions = model.predict(x_test_dict)

    # Convert symbols_test to list if it's a Series
    symbols_list = symbols_test.tolist() if hasattr(symbols_test, "tolist") else list(symbols_test)

    # Group by symbol
    unique_symbols = sorted(list(set(symbols_list)))

    print(f"\nEvaluating {len(unique_symbols)} stocks...")

    results = []
    detailed_predictions = []  # Store predictions for visualization

    for symbol in unique_symbols:
        # Get indices for this symbol
        symbol_mask = [s == symbol for s in symbols_list]
        symbol_indices = [i for i, mask in enumerate(symbol_mask) if mask]

        if len(symbol_indices) == 0:
            continue

        # Extract data for this symbol
        y_true_symbol = y_test.iloc[symbol_indices]
        y_pred_symbol = predictions[symbol_indices]
        dates_symbol = dates_test.iloc[symbol_indices]
        prices_symbol = prices_test.iloc[symbol_indices]

        # Calculate metrics
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

        mse = mean_squared_error(y_true_symbol, y_pred_symbol)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true_symbol, y_pred_symbol)
        r2 = r2_score(y_true_symbol, y_pred_symbol)

        # Calculate percentage-based metrics using actual prices
        percentage_errors = np.abs((y_pred_symbol - y_true_symbol) / y_true_symbol) * 100
        mape = percentage_errors.mean()  # Mean Absolute Percentage Error

        # Calculate directional accuracy (did we predict up/down correctly?)
        direction_actual = np.sign(y_true_symbol)
        direction_pred = np.sign(y_pred_symbol)
        directional_accuracy = (direction_actual == direction_pred).mean() * 100

        # Get test period date range
        start_date = dates_symbol.min()
        end_date = dates_symbol.max()

        results.append(
            {
                "symbol": symbol,
                "samples": len(symbol_indices),
                "rmse": rmse,
                "mae": mae,
                "mape": mape,
                "r2": r2,
                "directional_accuracy": directional_accuracy,
                "test_start": start_date,
                "test_end": end_date,
                "avg_price": prices_symbol.mean(),
            }
        )

        # Store detailed predictions for later visualization
        for i, idx in enumerate(symbol_indices):
            detailed_predictions.append(
                {
                    "symbol": symbol,
                    "date": dates_symbol.iloc[i],
                    "actual_return": y_true_symbol.iloc[i],
                    "predicted_return": y_pred_symbol[i],
                    "price": prices_symbol.iloc[i],
                }
            )

        print(f"\n{symbol}:")
        print(f"  Test Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"  Samples: {len(symbol_indices)}")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  MAE: {mae:.6f}")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  R²: {r2:.6f}")
        print(f"  Directional Accuracy: {directional_accuracy:.2f}%")

    # Summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics Across All Stocks")
    print("=" * 60)

    results_df = pd.DataFrame(results)
    print(f"\nNumber of stocks evaluated: {len(results_df)}")
    print(f"Average RMSE: {results_df['rmse'].mean():.6f}")
    print(f"Average MAE: {results_df['mae'].mean():.6f}")
    print(f"Average MAPE: {results_df['mape'].mean():.2f}%")
    print(f"Average R²: {results_df['r2'].mean():.6f}")
    print(f"Average Directional Accuracy: {results_df['directional_accuracy'].mean():.2f}%")
    print(
        f"\nBest performing stock (by R²): {results_df.loc[results_df['r2'].idxmax(), 'symbol']} (R²: {results_df['r2'].max():.6f})"
    )
    print(
        f"Worst performing stock (by R²): {results_df.loc[results_df['r2'].idxmin(), 'symbol']} (R²: {results_df['r2'].min():.6f})"
    )
    print(
        f"Best directional accuracy: {results_df.loc[results_df['directional_accuracy'].idxmax(), 'symbol']} ({results_df['directional_accuracy'].max():.2f}%)"
    )

    # Save results
    results_df.to_csv("results/per_stock_performance.csv", index=False)
    print(f"\n✓ Per-stock results saved to: results/per_stock_performance.csv")

    # Save detailed predictions for visualization
    detailed_df = pd.DataFrame(detailed_predictions)
    detailed_df.to_csv("results/detailed_predictions.csv", index=False)
    print(f"✓ Detailed predictions saved to: results/detailed_predictions.csv")
