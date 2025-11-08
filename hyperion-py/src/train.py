from typing import Any

from src.data import StockDataDownloader
from src.feature import create_target_features, FeatureEngineering
from src.writer import save_trained_model, persist_results, output_best_strategy
from src.xbg import XGBoostStockPredictor
from src.visualisation import generate_plots, Visualizer
from src.simulation import simulate_directional_trading_strategy, simulate_adaptive_threshold_strategy, \
    simulate_hold_days_strategy

TEST_SIZE = 0.3  # Train/test split ratio
USE_WALK_FORWARD = False  # Set to False for a simple train / test split


def simple_train_test_split(x, dates, prices, y) -> tuple[
    XGBoostStockPredictor, dict[str, float | Any], Any, Any, Any, Any, Any]:
    print("\n" + "=" * 60)
    print("Using Simple Train/Test Split")
    print("=" * 60)

    # Step 4: Train/test split
    split_idx = int(len(x) * (1 - TEST_SIZE))
    x_train, x_test = x.iloc[:split_idx], x.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    dates_test = dates[split_idx:]
    prices_test = prices[split_idx:]

    print(f"✓ Train set: {len(x_train)} samples")
    print(f"✓ Test set: {len(x_test)} samples")

    # Step 5: Train model
    predictor = XGBoostStockPredictor()
    predictor.train(x_train, y_train, x_test, y_test)

    # Step 6: Evaluate
    predictor.evaluate(x_train, y_train, "Train")
    test_results = predictor.evaluate(x_test, y_test, "Test")
    return x_test, x_train, dates_test, predictor, prices_test, test_results, y_test


def train_model(symbols=None, period: str = '5y', interval: str = '1d', visualization: bool = False):
    """
    Train a model for each ticker, defaults to AAPL
    :param visualization: should or should not generate plots
    :param symbols: List of stock symbols to process
    :param period: a Data period to download
    :param interval: a Data interval to download
    """
    if symbols is None:
        symbols = ['AAPL', 'NFLX']
    print("\n" + "=" * 60)
    print("XGBoost Stock Price Prediction & Trading Simulation")
    print("=" * 60)

    # Step 1: Download data
    downloader = StockDataDownloader(symbols, period=period, interval=interval)
    stock_data: dict[Any, Any] = downloader.download_data()

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
            df_features = create_target_features(df)

            # Step 3: Prepare features
            x, y, dates, prices, feature_columns = FeatureEngineering.prepare_features(df_features)
            print(f"✓ Prepared {len(x)} samples with {len(feature_columns)} features")

            x_test, x_train, dates_test, predictor, prices_test, test_results, y_test = simple_train_test_split(x,
                                                                                                                dates,
                                                                                                                prices,
                                                                                                                y)

            save_trained_model(predictor, symbol, test_results)

            if visualization: generate_plots(dates_test, df_features, predictor, symbol, test_results, y_test)

            # Step 8: Trading simulation with multiple strategies
            print("\n" + "=" * 60)
            print("Testing Multiple Trading Strategies")
            print("=" * 60)

            directional_trading_results, directional_simulator = simulate_directional_trading_strategy(dates_test,
                                                                                                       prices_test,
                                                                                                       test_results,
                                                                                                       y_test)

            adaptive_threshold_results, adaptive_simulator = simulate_adaptive_threshold_strategy(dates_test,
                                                                                                  prices_test,
                                                                                                  test_results,
                                                                                                  y_test)

            hold_days_results, hold_days_simulator = simulate_hold_days_strategy(dates_test, prices_test, test_results,
                                                                                 y_test)

            # Compare strategies and use the best performing one for plots
            strategies = [
                ('Directional', directional_trading_results, directional_simulator),
                ('Adaptive Threshold', adaptive_threshold_results, adaptive_simulator),
                ('Hold Days', hold_days_results, hold_days_simulator)
            ]

            # Find strategy with most trades (or best return if tied)
            valid_strategies = [(name, res, sim) for name, res, sim in strategies if res['num_trades'] > 0]

            if valid_strategies:
                best_strategy, sim_results = output_best_strategy(valid_strategies)
            else:
                print(f"\n⚠️  WARNING: No strategy generated trades!")
                sim_results = directional_trading_results  # Use first strategy anyway for plotting

            # Step 9: Plot trading results
            if visualization: Visualizer.plot_trading_simulation(sim_results, symbol)

            persist_results(x, x_test, x_train, best_strategy, period, sim_results, strategies, symbol, test_results,
                            valid_strategies)

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
    print(f"✓ Models saved in: models/")
    print("=" * 60 + "\n")
