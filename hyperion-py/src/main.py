import os
import warnings

import pandas as pd

from experimental import WalkForwardValidator
from feature import FeatureEngineering
from simulation import TradingSimulator, predict_today
from visualisation import Visualizer
from xbg import XGBoostStockPredictor
from data import StockDataDownloader

warnings.filterwarnings('ignore')

# Create directories for outputs
os.makedirs('historic_data', exist_ok=True)
os.makedirs('plots', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('models', exist_ok=True)


def main():
    """Main execution function"""
    print("\n" + "=" * 60)
    print("XGBoost Stock Price Prediction & Trading Simulation")
    print("=" * 60)

    # Configuration
    SYMBOLS = [
        'AAPL',
        # "GOOGL",
        # "TSLA",
        # "MSFT",
        "NFLX",
        # "NVDA",
        # "EXC",
        # "AMZN",
        # "CNM",
        # "COIN",
        # "AMGN",
        # "PLTR",
        # "PEP",
        # "META",
        # "ORCL",
        # "CRM",
        # "ADBE",
        # "AMD",
        # "AVGO",
        # "AMAT",
        # "NET",
        # "XZEM",
        # "BBAI",
        # "DLB",
        # "FMAO",
        # "PG",
        # "LMT",
        # "DELL",
        # "RKLB",
        # "CELH",
        # "FDX",
        # "BLK",
        # "IBM",
        # "HUT",
        # "MARA",
    ]  # Stock symbols to analyze
    PERIOD = '5y'  # Data period
    INTERVAL = '1d'  # Data interval
    TEST_SIZE = 0.3  # Train/test split ratio
    # PREDICTION_THRESHOLD = 0.002  # 0.2% threshold for trading signals

    # Walk-forward configuration
    USE_WALK_FORWARD = False  # Set to False for simple train/test split
    WALK_FORWARD_TRAIN_WINDOW = 180  # ~9 months of training data (reduced from 252)
    WALK_FORWARD_TEST_WINDOW = 20  # ~1 month of testing
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

            # Save the trained model
            print("\n" + "=" * 60)
            print("Saving Model")
            print("=" * 60)
            predictor.save_model(symbol)

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
                print(f"   Return: {best_strategy[1]['total_return'] * 100:.2f}%")
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
                    f.write(f"    Total Return: {strategy_results['total_return'] * 100:.2f}%\n")
                    f.write(f"    Number of Trades: {strategy_results['num_trades']}\n")
                    if strategy_results['num_trades'] > 0:
                        alpha = (strategy_results['total_return'] - strategy_results['buy_hold_return']) * 100
                        f.write(f"    Alpha vs Buy&Hold: {alpha:.2f}%\n")
                    f.write("\n")

                f.write(f"  Buy & Hold Return: {sim_results['buy_hold_return'] * 100:.2f}%\n")

                if valid_strategies:
                    f.write(f"\n  Best Strategy: {best_strategy[0]}\n")
                    f.write(f"    Final Value: ${best_strategy[1]['final_value']:,.2f}\n")
                    f.write(f"    Total Return: {best_strategy[1]['total_return'] * 100:.2f}%\n")
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
    print(f"✓ Models saved in: models/")
    print("=" * 60 + "\n")


def predict_mode():
    """Run prediction mode for today's data"""
    print("\n" + "=" * 80)
    print("PREDICTION MODE - Make predictions on today's data")
    print("=" * 80)

    # Get symbols from saved models
    model_files = [f for f in os.listdir('models') if f.endswith('_xgboost_model.pkl')]

    if not model_files:
        print("\n⚠️  No trained models found in 'models/' directory")
        print("   Please run training mode first to train models")
        return

    symbols = [f.replace('_xgboost_model.pkl', '') for f in model_files]

    print(f"\nFound trained models for: {', '.join(symbols)}")
    print("\nMaking predictions for all symbols...\n")

    predictions = []
    for symbol in symbols:
        result = predict_today(symbol)
        if result:
            predictions.append(result)

    # Summary
    if predictions:
        print("\n" + "=" * 80)
        print("SUMMARY OF ALL PREDICTIONS")
        print("=" * 80)
        print(f"\n{'Symbol':<10} {'Current':<12} {'Predicted':<12} {'Return':<10} {'Signal':<20}")
        print("-" * 80)
        for p in predictions:
            print(f"{p['symbol']:<10} ${p['current_price']:<11.2f} ${p['predicted_price']:<11.2f} "
                  f"{p['predicted_return'] * 100:>+8.3f}% {p['signal']:<20}")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
    predict_mode()

    # # Check for command line argument
    # if len(sys.argv) > 1 and sys.argv[1] == 'predict':
    #     # Prediction mode - use saved models on today's data
    #     predict_mode()
    # else:
    #     # Training mode - train new models
    #     main()
