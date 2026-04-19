import argparse
import os
import warnings

from src.config import HyperionConfig
from src.pipeline.stacked_pipeline import StackedModelTrainingPipeline
from src.pipeline.time_series_stacked_pipeline import TimeSeriesStackedModelTrainingPipeline

warnings.filterwarnings("ignore")

os.makedirs("./historic_data", exist_ok=True)
os.makedirs("./plots", exist_ok=True)
os.makedirs("./results", exist_ok=True)
os.makedirs("./models", exist_ok=True)
os.makedirs("./invalid_models", exist_ok=True)
os.makedirs("./params", exist_ok=True)


def _parse_args() -> HyperionConfig:
    defaults = HyperionConfig()
    parser = argparse.ArgumentParser(
        description="Hyperion — stock trading prediction system",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--period",
        default=defaults.period,
        help="Historical data period to download (e.g. '1y', '2y', '5y').",
    )
    parser.add_argument(
        "--intervals",
        default=",".join(defaults.intervals),
        help="Comma-separated data intervals (e.g. '1d,1h').",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=defaults.test_size,
        help="Fraction of data reserved for testing (0 < test_size < 1).",
    )
    parser.add_argument(
        "--target-days",
        type=int,
        default=defaults.target_days,
        help="Number of forward days used to compute the return target.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=defaults.n_trials,
        help="Number of Optuna hyperparameter-search trials per model.",
    )
    parser.add_argument(
        "--r2-save-threshold",
        type=float,
        default=defaults.r2_save_threshold,
        help="Minimum R² required to persist a model to disk.",
    )
    parser.add_argument(
        "--r2-invalid-threshold",
        type=float,
        default=defaults.r2_invalid_threshold,
        help="R² below this value causes the model to be saved as invalid.",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=defaults.initial_capital,
        help="Starting cash balance for trading simulations.",
    )
    parser.add_argument(
        "--transaction-cost",
        type=float,
        default=defaults.transaction_cost,
        help="Proportional transaction cost per trade (e.g. 0.001 = 0.1 %%).",
    )
    args = parser.parse_args()
    return HyperionConfig(
        period=args.period,
        intervals=[i.strip() for i in args.intervals.split(",")],
        test_size=args.test_size,
        target_days=args.target_days,
        n_trials=args.n_trials,
        r2_save_threshold=args.r2_save_threshold,
        r2_invalid_threshold=args.r2_invalid_threshold,
        initial_capital=args.initial_capital,
        transaction_cost=args.transaction_cost,
    )


if __name__ == "__main__":
    config = _parse_args()

    stacked_pipeline: StackedModelTrainingPipeline = StackedModelTrainingPipeline(
        period=config.period,
        intervals=config.intervals,
        test_size=config.test_size,
        n_trials=config.n_trials,
        target_days=config.target_days,
        r2_save_threshold=config.r2_save_threshold,
        r2_invalid_threshold=config.r2_invalid_threshold,
    )

    (
        stacked_pipeline.read_tickers()
        .download_data()
        .prepare_features()
        .train()
        .evaluate_model()
        .simulate(initial_capital=config.initial_capital, transaction_cost=config.transaction_cost)
    )

    time_series_stacked_pipeline: TimeSeriesStackedModelTrainingPipeline = TimeSeriesStackedModelTrainingPipeline(
        period=config.period,
        interval=config.intervals[0],
        test_size=config.test_size,
        n_trials=config.n_trials,
        target_days=config.target_days,
        r2_save_threshold=config.r2_save_threshold,
        r2_invalid_threshold=config.r2_invalid_threshold,
    )

    (
        time_series_stacked_pipeline.read_tickers()
        .download_data()
        .prepare_features()
        .train()
        .analyze_stacker_performance()
    )
