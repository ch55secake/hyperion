# Hyperion

> [!NOTE]
> This project is under active development.

## Overview

Hyperion is a stock trading prediction system that trains machine-learning models on historical price data and uses them to simulate trades. The target return on the portfolio is **5-10% ROI**.

The system downloads data for 634 tickers via [yfinance](https://github.com/ranaroussi/yfinance), engineers 100+ technical indicator features, trains a stacked XGBoost/LightGBM ensemble, and evaluates predictions through a multi-strategy trading simulator.

## Features

- **Model training** -- XGBoost and LightGBM stacked ensemble with optional CatBoost
- **Multi-timeframe** -- Daily and hourly data combined in a stacked pipeline
- **Feature engineering** -- 100+ technical indicators (SMA, EMA, WMA, HMA, RSI, MACD, Bollinger Bands, CCI, ATR, and more)
- **Hyperparameter optimisation** -- Automated search via [Optuna](https://optuna.org/)
- **Trading simulation** -- 13 strategies with configurable thresholds and transaction costs
- **Walk-forward validation** -- Experimental time-series-aware evaluation
- **Visualisation** -- Prediction plots, trading simulation charts, feature correlation heatmaps
- **Flask API server** -- Endpoints for predictions and training

## Trading Strategies

| Strategy | Description |
|----------|-------------|
| Directional | Trades in the direction of the predicted return |
| Adaptive | Adjusts threshold dynamically based on recent predictions |
| Hold Days | Holds positions for a fixed number of days |
| Stop-Loss / Take-Profit | Exits on configurable SL/TP levels |
| Momentum | Trades based on momentum signals |
| Contrarian | Trades against the predicted direction |
| Bollinger Band Reversion | Mean-reversion using Bollinger Bands |
| EMA Cross | Trades on EMA crossover signals |
| SMA Trend | Follows SMA-based trend signals |
| Time Stop | Exits after a maximum holding period |
| Hybrid Trend ML | Combines trend-following with ML predictions |
| Volatility Adjusted | Adjusts thresholds based on volatility |
| Coinflip | Random baseline for strategy comparison |

## Repository Layout

```
hyperion/
├── src/
│   ├── align/          # Target alignment across timeframes
│   ├── data/           # Data fetching and caching (yfinance)
│   ├── experimental/   # Walk-forward validation
│   ├── feature/        # Feature engineering and technical indicators
│   ├── model/          # Model implementations
│   │   ├── xbg/        #   XGBoost predictor
│   │   ├── lgb/        #   LightGBM predictor
│   │   ├── catboost/   #   CatBoost predictor
│   │   └── stacker/    #   Weighted ensemble / time-series stacker
│   ├── optimise/       # Optuna hyperparameter optimisation
│   ├── pipeline/       # End-to-end training and prediction pipelines
│   ├── server/         # Flask API server
│   ├── simulation/     # Trading simulator and strategies
│   ├── util/           # Shared utilities (logger singleton)
│   ├── visualisation/  # Plots and charts
│   ├── writer/         # Results and model persistence
│   └── main.py         # CLI entry point
├── tests/              # Unit tests (pytest)
├── resources/          # Ticker lists
├── pyproject.toml      # uv project definition
├── Makefile            # Developer commands
└── .pre-commit-config.yaml
```

## Getting Started

### Prerequisites

- Python **3.12**
- [uv](https://docs.astral.sh/uv/)

### Installation

```bash
make install    # uv sync
```

### Install pre-commit hooks

```bash
pre-commit install
```

### Running

```bash
make run        # uv run python3 src/main.py (uses all defaults)
```

All parameters have sensible defaults but can be overridden via Make variables or `ARGS`:

```bash
# Override individual parameters
make run PERIOD=5y N_TRIALS=200

# Override multiple parameters
make run PERIOD=5y TEST_SIZE=0.3 VAL_SIZE=0.1 INITIAL_CAPITAL=50000

# Pass arbitrary CLI flags directly
make run ARGS="--period 5y --n-trials 200 --transaction-cost 0.002"

# See all available parameters and their defaults
make help
```

| Make variable | CLI flag | Default | Purpose |
|---|---|---|---|
| `PERIOD` | `--period` | `2y` | Historical data window |
| `INTERVALS` | `--intervals` | `1d,1h` | OHLCV intervals (comma-separated) |
| `TEST_SIZE` | `--test-size` | `0.2` | Train/test split fraction |
| `VAL_SIZE` | `--val-size` | `0.1` | Validation split fraction (from training window) |
| `TARGET_DAYS` | `--target-days` | `10` | Forward-return horizon (days) |
| `N_TRIALS` | `--n-trials` | `1000` | Optuna trials per model |
| `R2_SAVE` | `--r2-save-threshold` | `0.0012` | Min R² to persist a model |
| `R2_INVALID` | `--r2-invalid-threshold` | `-0.3` | R² floor for invalid-model path |
| `INITIAL_CAPITAL` | `--initial-capital` | `10000` | Simulation starting cash |
| `TRANSACTION_COST` | `--transaction-cost` | `0.001` | Per-trade proportional cost |

## Development

### Code Style

- **Formatter / Linter:** [Ruff](https://docs.astral.sh/ruff/) -- line length 120 (replaces Black and Pylint)
- **Type checker:** [ty](https://docs.astral.sh/ty/) -- fast type checking from the Astral ecosystem
- **Pre-commit hooks** run automatically on commit and push

### Testing

```bash
make test       # uv run pytest tests/ -v --tb=short
make test-cov   # run tests with coverage report
```

### Makefile Commands

| Command | Effect |
|---------|--------|
| `make install` | Install dependencies via uv |
| `make run` | Run the main pipeline with default parameters |
| `make run PERIOD=5y N_TRIALS=200` | Run with overridden parameters |
| `make help` | Show all CLI parameters and their defaults |
| `make test` | Run unit tests |
| `make test-cov` | Run tests with coverage report |
| `make clean` | Remove plots, invalid models, results, and params |
| `make cleanmodels` | Remove plots, all models, invalid models, results, and params |
| `make ctrain` | Clean then run |
| `make cmtrain` | Clean models then run |

### CI Pipeline

GitHub Actions runs the following checks on every push and PR to `main`:

| Job | Description |
|-----|-------------|
| **ruff** | Linting and formatting check with ruff |
| **build** | Dependency installation smoke test |
| **test** | Unit tests with coverage reporting |

