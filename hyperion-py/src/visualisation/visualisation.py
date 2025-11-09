from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import Series

from src.xbg import XGBoostStockPredictor


class Visualizer:
    """Creates visualizations for predictions and trading results"""

    @staticmethod
    def plot_predictions(dates, actual, predicted, symbol, save_path="plots"):
        """Plot actual vs predicted returns"""
        plt.figure(figsize=(15, 6))

        plt.subplot(1, 2, 1)
        plt.scatter(actual, predicted, alpha=0.5)
        plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], "r--", lw=2)
        plt.xlabel("Actual Returns")
        plt.ylabel("Predicted Returns")
        plt.title(f"{symbol} - Actual vs Predicted Returns")
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.plot(dates, actual, label="Actual", alpha=0.7)
        plt.plot(dates, predicted, label="Predicted", alpha=0.7)
        plt.xlabel("Date")
        plt.ylabel("Returns")
        plt.title(f"{symbol} - Returns Over Time")
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{save_path}/{symbol}_predictions.png", dpi=300, bbox_inches="tight")
        print(f"  ✓ Saved predictions plot: {save_path}/{symbol}_predictions.png")
        plt.close()

    @staticmethod
    def plot_feature_importance(feature_importance, symbol, top_n=15, save_path="plots"):
        """Plot feature importance"""
        plt.figure(figsize=(10, 8))

        top_features = feature_importance.head(top_n)
        plt.barh(range(len(top_features)), top_features["importance"])
        plt.yticks(range(len(top_features)), top_features["feature"])
        plt.xlabel("Importance")
        plt.title(f"{symbol} - Top {top_n} Feature Importance")
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()
        plt.savefig(f"{save_path}/{symbol}_feature_importance.png", dpi=300, bbox_inches="tight")
        print(f"  ✓ Saved feature importance plot: {save_path}/{symbol}_feature_importance.png")
        plt.close()

    @staticmethod
    def plot_trading_simulation(simulation_results, symbol, save_path="plots"):
        """Plot trading simulation results"""
        portfolio_df = simulation_results["portfolio_history"]
        trades_df = simulation_results["trades"]

        fig, axes = plt.subplots(3, 1, figsize=(15, 14))

        # Portfolio value over time
        ax1 = axes[0]
        ax1.plot(
            portfolio_df["date"], portfolio_df["portfolio_value"], label="Portfolio Value", linewidth=2, color="blue"
        )
        ax1.axhline(
            y=simulation_results["final_value"],
            color="g",
            linestyle="--",
            alpha=0.5,
            label=f"Final: ${simulation_results['final_value']:,.0f}",
        )

        # Show buy & hold for comparison
        first_price = portfolio_df["price"].iloc[0]
        buy_hold_values = (portfolio_df["price"] / first_price) * simulation_results["final_value"]
        ax1.plot(
            portfolio_df["date"],
            buy_hold_values,
            label="Buy & Hold",
            linewidth=2,
            alpha=0.5,
            linestyle="--",
            color="orange",
        )

        ax1.set_xlabel("Date")
        ax1.set_ylabel("Portfolio Value ($)")
        ax1.set_title(f"{symbol} - Portfolio Value Over Time (Trades: {len(trades_df)})")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis="x", rotation=45)

        # Stock price with buy/sell signals
        ax2 = axes[1]
        ax2.plot(
            portfolio_df["date"], portfolio_df["price"], label="Stock Price", linewidth=2, alpha=0.7, color="black"
        )

        if len(trades_df) > 0:
            buys = trades_df[trades_df["action"].str.contains("BUY", case=False, na=False)]
            sells = trades_df[trades_df["action"].str.contains("SELL", case=False, na=False)]

            if len(buys) > 0:
                ax2.scatter(
                    buys["date"],
                    buys["price"],
                    color="green",
                    marker="^",
                    s=150,
                    label=f"Buy ({len(buys)})",
                    zorder=5,
                    edgecolors="black",
                    linewidth=1,
                )
            if len(sells) > 0:
                ax2.scatter(
                    sells["date"],
                    sells["price"],
                    color="red",
                    marker="v",
                    s=150,
                    label=f"Sell ({len(sells)})",
                    zorder=5,
                    edgecolors="black",
                    linewidth=1,
                )
        else:
            # Add warning text if no trades
            ax2.text(
                0.5,
                0.5,
                "⚠️ NO TRADES EXECUTED",
                transform=ax2.transAxes,
                ha="center",
                va="center",
                fontsize=16,
                color="red",
                alpha=0.5,
                fontweight="bold",
                bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.3),
            )

        ax2.set_xlabel("Date")
        ax2.set_ylabel("Stock Price ($)")
        ax2.set_title(f"{symbol} - Stock Price with Trading Signals")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis="x", rotation=45)

        # Predictions over time
        ax3 = axes[2]
        if "prediction" in portfolio_df.columns:
            predictions = portfolio_df["prediction"].values * 100  # Convert to percentage
            ax3.plot(
                portfolio_df["date"],
                predictions,
                label="Predicted Return (%)",
                linewidth=1.5,
                alpha=0.8,
                color="purple",
            )
            ax3.axhline(y=0, color="black", linestyle="-", linewidth=1, alpha=0.5)

            # Fill areas
            ax3.fill_between(
                portfolio_df["date"],
                0,
                predictions,
                where=(predictions > 0),
                alpha=0.3,
                color="green",
                label="Positive Prediction",
            )
            ax3.fill_between(
                portfolio_df["date"],
                0,
                predictions,
                where=(predictions < 0),
                alpha=0.3,
                color="red",
                label="Negative Prediction",
            )

            # Add statistics
            pred_stats = f"Mean: {np.mean(predictions):.3f}% | Std: {np.std(predictions):.3f}% | Range: [{np.min(predictions):.3f}, {np.max(predictions):.3f}]%"
            ax3.text(
                0.02,
                0.98,
                pred_stats,
                transform=ax3.transAxes,
                verticalalignment="top",
                fontsize=9,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

        ax3.set_xlabel("Date")
        ax3.set_ylabel("Predicted Return (%)")
        ax3.set_title(f"{symbol} - Model Predictions Over Time")
        ax3.legend(loc="upper right")
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.savefig(f"{save_path}/{symbol}_trading_simulation.png", dpi=300, bbox_inches="tight")
        print(f"  ✓ Saved trading simulation plot: {save_path}/{symbol}_trading_simulation.png")
        plt.close()

    @staticmethod
    def plot_walk_forward_results(wf_results, symbol, save_path="plots"):
        """Plot walk-forward analysis results with fold boundaries"""
        fig, axes = plt.subplots(3, 1, figsize=(16, 14))

        dates = wf_results["dates"]
        predictions = wf_results["predictions"]
        actuals = wf_results["actuals"]
        fold_boundaries = wf_results["fold_boundaries"]

        # Plot 1: Predictions vs Actuals with fold boundaries
        ax1 = axes[0]
        ax1.plot(dates, actuals, label="Actual Returns", alpha=0.7, linewidth=1.5)
        ax1.plot(dates, predictions, label="Predicted Returns", alpha=0.7, linewidth=1.5)
        ax1.axhline(y=0, color="black", linestyle="-", linewidth=0.5, alpha=0.3)

        # Mark retrain points
        for boundary in fold_boundaries:
            ax1.axvline(x=boundary["start_date"], color="red", linestyle="--", alpha=0.3, linewidth=1)

        # Add legend entry for retrain markers
        ax1.axvline(x=dates[0], color="red", linestyle="--", alpha=0.5, linewidth=1, label="Retrain Point")

        ax1.set_xlabel("Date")
        ax1.set_ylabel("Returns")
        ax1.set_title(f"{symbol} - Walk-Forward: Predictions vs Actuals ({len(fold_boundaries)} folds)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis="x", rotation=45)

        # Plot 2: Prediction errors over time
        ax2 = axes[1]
        errors = predictions - actuals
        ax2.plot(dates, errors, label="Prediction Error", alpha=0.7, linewidth=1, color="red")
        ax2.axhline(y=0, color="black", linestyle="-", linewidth=1)
        ax2.fill_between(dates, 0, errors, alpha=0.3, color="red")

        # Mark retrain points
        for boundary in fold_boundaries:
            ax2.axvline(x=boundary["start_date"], color="blue", linestyle="--", alpha=0.3, linewidth=1)

        # Add statistics
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors**2))
        stats_text = f"MAE: {mae:.6f} | RMSE: {rmse:.6f}"
        ax2.text(
            0.02,
            0.98,
            stats_text,
            transform=ax2.transAxes,
            verticalalignment="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        ax2.set_xlabel("Date")
        ax2.set_ylabel("Error")
        ax2.set_title(f"{symbol} - Prediction Errors Over Time")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis="x", rotation=45)

        # Plot 3: Cumulative returns comparison
        ax3 = axes[2]

        # Calculate cumulative returns
        actual_cumulative = (1 + pd.Series(actuals)).cumprod() - 1
        predicted_cumulative = (1 + pd.Series(predictions)).cumprod() - 1

        ax3.plot(dates, actual_cumulative * 100, label="Actual Cumulative Return", linewidth=2, alpha=0.8)
        ax3.plot(dates, predicted_cumulative * 100, label="Predicted Cumulative Return", linewidth=2, alpha=0.8)

        # Mark retrain points
        for boundary in fold_boundaries:
            ax3.axvline(x=boundary["start_date"], color="green", linestyle="--", alpha=0.3, linewidth=1)

        ax3.set_xlabel("Date")
        ax3.set_ylabel("Cumulative Return (%)")
        ax3.set_title(f"{symbol} - Cumulative Returns (Walk-Forward)")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.savefig(f"{save_path}/{symbol}_walk_forward.png", dpi=300, bbox_inches="tight")
        print(f"  ✓ Saved walk-forward plot: {save_path}/{symbol}_walk_forward.png")
        plt.close()

    @staticmethod
    def plot_forecast(historical_df, forecast_data, symbol, save_path="plots"):
        """Plot historical prices and forecasted prices"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

        # Plot 1: Price forecast
        hist_dates = historical_df.index
        hist_prices = historical_df["Close"].values

        forecast_dates = forecast_data["dates"]
        forecast_prices = forecast_data["prices"]
        forecast_lower = forecast_data["lower_bound"]
        forecast_upper = forecast_data["upper_bound"]

        # Historical prices
        ax1.plot(hist_dates, hist_prices, label="Historical Price", linewidth=2, color="blue", alpha=0.8)

        # Mark today
        ax1.axvline(x=hist_dates[-1], color="green", linestyle="--", linewidth=2, alpha=0.7, label="Today")

        # Forecast prices
        ax1.plot(
            forecast_dates,
            forecast_prices,
            label="Forecasted Price",
            linewidth=2,
            color="red",
            linestyle="--",
            alpha=0.8,
        )

        # Confidence interval
        ax1.fill_between(
            forecast_dates, forecast_lower, forecast_upper, alpha=0.3, color="red", label="Confidence Interval (±1 std)"
        )

        # Add start/end labels
        current_price = hist_prices[-1]
        final_forecast = forecast_prices[-1]
        ax1.scatter([hist_dates[-1]], [current_price], color="green", s=200, zorder=5, edgecolors="black", linewidth=2)
        ax1.scatter(
            [forecast_dates[-1]], [final_forecast], color="red", s=200, zorder=5, edgecolors="black", linewidth=2
        )

        ax1.text(
            hist_dates[-1],
            current_price,
            f"  Today\n  ${current_price:.2f}",
            verticalalignment="center",
            fontsize=10,
            fontweight="bold",
        )
        ax1.text(
            forecast_dates[-1],
            final_forecast,
            f"  180-day\n  ${final_forecast:.2f}",
            verticalalignment="center",
            fontsize=10,
            fontweight="bold",
        )

        ax1.set_xlabel("Date", fontsize=12)
        ax1.set_ylabel("Price ($)", fontsize=12)
        ax1.set_title(f"{symbol} - 180-Day Price Forecast", fontsize=14, fontweight="bold")
        ax1.legend(loc="best", fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis="x", rotation=45)

        # Plot 2: Daily returns forecast
        forecast_returns = forecast_data["daily_returns"]

        colors = ["green" if r > 0 else "red" for r in forecast_returns]
        ax2.bar(
            forecast_dates, np.array(forecast_returns) * 100, color=colors, alpha=0.6, edgecolor="black", linewidth=0.5
        )
        ax2.axhline(y=0, color="black", linestyle="-", linewidth=1)

        # Add cumulative return line
        cumulative_return = (1 + np.array(forecast_returns)).cumprod() - 1
        ax3 = ax2.twinx()
        ax3.plot(
            forecast_dates,
            cumulative_return * 100,
            color="blue",
            linewidth=2,
            marker="o",
            markersize=4,
            label="Cumulative Return",
        )

        total_return = cumulative_return[-1] * 100
        ax2.text(
            0.02,
            0.98,
            f"180-Day Expected Return: {total_return:+.2f}%",
            transform=ax2.transAxes,
            verticalalignment="top",
            fontsize=12,
            fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7),
        )

        ax2.set_xlabel("Date", fontsize=12)
        ax2.set_ylabel("Daily Return (%)", fontsize=12)
        ax3.set_ylabel("Cumulative Return (%)", fontsize=12, color="blue")
        ax3.tick_params(axis="y", labelcolor="blue")
        ax2.set_title(f"{symbol} - Forecasted Daily Returns", fontsize=14, fontweight="bold")
        ax2.grid(True, alpha=0.3, axis="y")
        ax2.tick_params(axis="x", rotation=45)
        ax3.legend(loc="upper left", fontsize=10)

        plt.tight_layout()
        filename = f"{save_path}/{symbol}_180day_forecast.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"  ✓ Saved forecast plot: {filename}")
        plt.close()

    @staticmethod
    def plot_price_with_indicators(df, symbol, save_path="plots"):
        """Plot stock price with technical indicators"""
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))

        # Price with moving averages
        ax1 = axes[0]
        ax1.plot(df.index, df["Close"], label="Close", linewidth=2)
        ax1.plot(df.index, df["SMA_20"], label="SMA 20", alpha=0.7)
        ax1.plot(df.index, df["SMA_50"], label="SMA 50", alpha=0.7)
        ax1.fill_between(df.index, df["BB_Lower"], df["BB_Upper"], alpha=0.2, label="Bollinger Bands")
        ax1.set_ylabel("Price ($)")
        ax1.set_title(f"{symbol} - Price with Moving Averages")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # RSI
        ax2 = axes[1]
        ax2.plot(df.index, df["RSI"], linewidth=2)
        ax2.axhline(y=70, color="r", linestyle="--", alpha=0.5)
        ax2.axhline(y=30, color="g", linestyle="--", alpha=0.5)
        ax2.set_ylabel("RSI")
        ax2.set_title(f"{symbol} - Relative Strength Index")
        ax2.set_ylim([0, 100])
        ax2.grid(True, alpha=0.3)

        # Volume
        ax3 = axes[2]
        ax3.bar(df.index, df["Volume"], alpha=0.7)
        ax3.set_xlabel("Date")
        ax3.set_ylabel("Volume")
        ax3.set_title(f"{symbol} - Trading Volume")
        ax3.grid(True, alpha=0.3)

        for ax in axes:
            ax.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.savefig(f"{save_path}/{symbol}_technical_indicators.png", dpi=300, bbox_inches="tight")
        print(f"  ✓ Saved technical indicators plot: {save_path}/{symbol}_technical_indicators.png")
        plt.close()


def combined_feature_importance(stacked_predictor):
    combined = None
    for name, model in stacked_predictor.models.items():
        if hasattr(model, "feature_importance") and model.feature_importance is not None:
            fi = model.feature_importance.copy()
            # Apply weight if desired
            weight = stacked_predictor.weights.get(name, 1.0)
            fi["importance"] *= weight
            fi.rename(columns={"importance": f"importance_{name}"}, inplace=True)
            if combined is None:
                combined = fi
            else:
                combined = combined.merge(fi, on="feature", how="outer")
    if combined is not None:
        # Sum weighted importances across models
        importance_cols = [c for c in combined.columns if c.startswith("importance_")]
        combined["importance"] = combined[importance_cols].sum(axis=1)
        return combined[["feature", "importance"]].sort_values("importance", ascending=False)
    return None


def generate_plots(
    dates_test: Any,
    df_features,
    predictor: dict[str, np.ndarray[Any, np.dtype[Any]] | list[Any] | dict[str, float | Any] | float | Any] | Any,
    symbol,
    test_results: Series | Any,
    y_test: Series | Any,
):
    # Step 7: Generate plots
    print("\nGenerating visualizations...")
    for name, model in predictor.models.items():
        if hasattr(model, "feature_importance") and model.feature_importance is not None:
            Visualizer.plot_feature_importance(model.feature_importance, f"{symbol}_{name}")
    fi_combined = combined_feature_importance(predictor)
    if fi_combined is not None:
        Visualizer.plot_feature_importance(fi_combined, symbol)
    # Plot predictions
    Visualizer.plot_predictions(dates_test, y_test, test_results["predictions"], symbol)

    # Plot feature importance (combined from stacked models)
    if predictor.feature_importance is not None:
        Visualizer.plot_feature_importance(predictor.feature_importance, symbol)

    # Plot price + indicators
    Visualizer.plot_price_with_indicators(df_features, symbol)


def generate_walk_forward_plots(wf_results, symbol):
    # Generate walk-forward specific plots
    print("\nGenerating walk-forward visualizations...")
    Visualizer.plot_walk_forward_results(wf_results, symbol)
