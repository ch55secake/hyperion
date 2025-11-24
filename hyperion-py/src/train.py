import json
import os
import traceback
from typing import Dict, Tuple

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
from src.simulation import TradingSimulator
from src.simulation.strategy.directional import DirectionalTradingStrategy
from src.simulation.strategy.adaptive import AdaptiveThresholdStrategy
from src.simulation.strategy.hold_days import HoldDaysStrategy
from src.simulation.strategy.strategy import Strategy

TEST_SIZE = 0.3  # Train/test split ratio
USE_WALK_FORWARD = False  # Set to False for a simple train / test split


def load_best_params_file(symbol: str, model_type: str):
    with open(f"params/{symbol}_best_params.json", "r", encoding="UTF-8") as f:
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
        with open("resources/tickers.txt", "r", encoding="UTF-8") as f:
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
