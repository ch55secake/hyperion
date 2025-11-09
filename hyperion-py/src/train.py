from typing import Any
import traceback

import numpy as np
import pandas as pd

from src.data import StockDataDownloader
from src.feature import create_target_features, FeatureEngineering
from src.writer import save_trained_model, persist_results, output_best_strategy
from src.xbg import XGBoostStockPredictor
from src.visualisation import generate_plots, Visualizer
from src.simulation import (
    simulate_directional_trading_strategy,
    simulate_adaptive_threshold_strategy,
    simulate_hold_days_strategy,
)

TEST_SIZE = 0.3  # Train/test split ratio
USE_WALK_FORWARD = False  # Set to False for a simple train / test split


def simple_train_test_split(
    x, dates, prices, y
) -> tuple[XGBoostStockPredictor, dict[str, float | Any], Any, Any, Any, Any, Any]:
    """
    Simple train/test split with proper time-series handling
    """
    print("\n" + "=" * 60)
    print("Using Simple Train/Test Split")
    print("=" * 60)

    # Debug: Check data before split
    print("\nData Quality Check:")
    print(f"Total samples: {len(x)}")
    print(f"Features shape: {x.shape}")
    print(f"Target shape: {y.shape}")
    print(f"NaN in features: {x.isna().sum().sum()}")
    print(f"NaN in target: {y.isna().sum()}")

    # Check for data leakage - FIXED
    print("\nData Leakage Check:")
    print(f"First date: {dates[0]}")  # ✓ Direct indexing
    print(f"Last date: {dates[-1]}")  # ✓ Direct indexing
    print(f"Target mean: {y.mean():.6f} (should be ~0)")
    print(f"Target std: {y.std():.6f} (should be 0.01-0.03)")

    # Show alignment - FIXED
    print("\nFirst 3 rows (verify alignment):")
    sample_df = pd.DataFrame(
        {
            "Date": dates[:3],  # ✓ Slice instead of iloc
            "Price": prices.iloc[:3],
            "SMA_20": x["SMA_20"].iloc[:3],
            "Target": y.iloc[:3],
        }
    )
    print(sample_df)

    # Step 4: Train/test split (time-based)
    split_idx = int(len(x) * (1 - TEST_SIZE))

    x_train, x_test = x.iloc[:split_idx], x.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    dates_test = dates[split_idx:]  # ✓ Direct slicing
    prices_test = prices.iloc[split_idx:]

    print(f"\n✓ Train set: {len(x_train)} samples")
    print(f"  Date range: {dates[0]} to {dates[split_idx-1]}")  # ✓ Fixed
    print(f"✓ Test set: {len(x_test)} samples")
    print(f"  Date range: {dates[split_idx]} to {dates[-1]}")  # ✓ Fixed

    # Verify no overlap - FIXED
    assert dates[split_idx - 1] < dates[split_idx], "Train/test dates overlap!"

    # Step 5: Train model
    print("\nTraining XGBoost model...")
    predictor = XGBoostStockPredictor()
    predictor.train(x_train, y_train, x_test, y_test)

    # Step 6: Evaluate
    print("\n" + "=" * 60)
    print("Model Evaluation")
    print("=" * 60)

    predictor.evaluate(x_train, y_train, "Train")
    test_results = predictor.evaluate(x_test, y_test, "Test")

    # Additional diagnostics
    print("\nDiagnostics:")
    y_train_pred = predictor.predict(x_train)
    y_test_pred = predictor.predict(x_test)

    # Check if predictions are reasonable
    print(f"Train predictions - mean: {y_train_pred.mean():.6f}, std: {y_train_pred.std():.6f}")
    print(f"Test predictions - mean: {y_test_pred.mean():.6f}, std: {y_test_pred.std():.6f}")

    # Directional accuracy
    train_dir_acc = (np.sign(y_train_pred) == np.sign(y_train)).mean() * 100
    test_dir_acc = (np.sign(y_test_pred) == np.sign(y_test)).mean() * 100
    print(f"Train directional accuracy: {train_dir_acc:.2f}%")
    print(f"Test directional accuracy: {test_dir_acc:.2f}%")

    return x_test, x_train, dates_test, predictor, prices_test, test_results, y_test


def train_model(symbols=None, period: str = "5y", interval: str = "1d", visualization: bool = False):
    """
    Train a model for each ticker, defaults to AAPL
    :param visualization: should or should not generate plots
    :param symbols: List of stock symbols to process
    :param period: a Data period to download
    :param interval: a Data interval to download
    """
    if symbols is None:
        symbols = [
            "AAPL",
            "NFLX",
            "TSLA",
            "MSFT",
            "GOOG",
            "AMZN",
            "META",
            "SPY",
            "VOO",
            "VTI",
            "VOOG",
            "VOOD",
            "VOOX",
            "EXC",
            "DELL",
            "AMD",
            "AMGN",
            "PEP",
            "PLTR",
            "CNM",
            "BMNR",
            "COIN",
            "MARA",
            "HUT",
            "IBM",
            "BA",
            "FDX",
            "LMT",
            "NET",
            "FMAO",
        ]
        # symbols = ["AAPL"]
    print("\n" + "=" * 60)
    print("XGBoost Stock Price Prediction & Trading Simulation")
    print("=" * 60)

    # Step 1: Download data
    downloader = StockDataDownloader(symbols, period=period, interval=interval)
    stock_data: dict[Any, Any] = downloader.download_data()

    if not stock_data:
        print("\n⚠️  No data downloaded. Exiting.")
        return False

    # Process each symbol
    for symbol, df in stock_data.items():
        try:
            print("\n" + "=" * 60)
            print(f"Processing {symbol}")
            print("=" * 60)

            # Step 2: Feature engineering
            print("\nAdding technical indicators...")
            df_features = create_target_features(df)

            # Step 3: Prepare features
            x, y, dates, prices, feature_columns = FeatureEngineering.prepare_features(df_features)
            print(f"✓ Prepared {len(x)} samples with {len(feature_columns)} features")

            x_test, x_train, dates_test, predictor, prices_test, test_results, y_test = simple_train_test_split(
                x, dates, prices, y
            )

            save_trained_model(predictor, symbol, test_results)

            if visualization:
                generate_plots(dates_test, df_features, predictor, symbol, test_results, y_test)

            run_trade_simulation(
                dates_test, period, prices_test, symbol, test_results, visualization, x, x_test, x_train, y_test
            )

        except Exception as e:
            print(f"\n✗ Error processing {symbol}: {str(e)}")
            traceback.print_exc()
            return False

    print("\n" + "=" * 60)
    print("✓ All processing complete!")
    print("✓ Plots saved in: plots/")
    print("✓ Results saved in: results/")
    print("✓ CSV data saved in: historic_data/")
    print("✓ Models saved in: models/")
    print("=" * 60 + "\n")

    return True


def run_trade_simulation(
    dates_test,
    period: str,
    prices_test,
    symbol,
    test_results,
    visualization: bool,
    x,
    x_test: XGBoostStockPredictor,
    x_train: dict[str, float | Any],
    y_test,
):
    print("\n" + "=" * 60)
    print("Testing Multiple Trading Strategies")
    print("=" * 60)

    directional_trading_results, directional_simulator = simulate_directional_trading_strategy(
        dates_test, prices_test, test_results, y_test
    )

    adaptive_threshold_results, adaptive_simulator = simulate_adaptive_threshold_strategy(
        dates_test, prices_test, test_results, y_test
    )

    hold_days_results, hold_days_simulator = simulate_hold_days_strategy(dates_test, prices_test, test_results, y_test)

    # Compare strategies and use the best performing one for plots
    strategies = [
        ("Directional", directional_trading_results, directional_simulator),
        ("Adaptive Threshold", adaptive_threshold_results, adaptive_simulator),
        ("Hold Days", hold_days_results, hold_days_simulator),
    ]

    # Find strategy with most trades (or best return if tied)
    valid_strategies = [(name, res, sim) for name, res, sim in strategies if res["num_trades"] > 0]

    if valid_strategies:
        best_strategy, sim_results = output_best_strategy(valid_strategies)
        if best_strategy is not None:
            persist_results(
                x,
                x_test,
                x_train,
                best_strategy,
                period,
                sim_results,
                strategies,
                symbol,
                test_results,
                valid_strategies,
            )

    else:
        print("\n⚠️  WARNING: No strategy generated trades!")
        sim_results = directional_trading_results  # Use first strategy anyway for plotting

    if visualization:
        Visualizer.plot_trading_simulation(sim_results, symbol)
