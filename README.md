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
│   ├── main.py         # CLI entry point
│   └── train.py        # Training entry point
├── tests/              # Unit tests (pytest)
├── resources/          # Ticker lists
├── pyproject.toml      # Poetry project definition
├── Makefile            # Developer commands
└── .pre-commit-config.yaml
```

## Getting Started

### Prerequisites

- Python **3.12**
- [Poetry](https://python-poetry.org/)

### Installation

```bash
make install    # poetry lock && poetry install
```

### Install pre-commit hooks

```bash
pre-commit install
```

### Running

```bash
make run        # poetry run python3 src/main.py
```

## Development

### Code Style

- **Formatter:** [Black](https://black.readthedocs.io/) -- line length 120
- **Linter:** [Pylint](https://pylint.org/) -- minimum score 9.0, max line length 120
- **Pre-commit hooks** run automatically on commit and push

### Testing

```bash
make test       # poetry run pytest tests/ -v --tb=short
make test-cov   # run tests with coverage report
```

### Makefile Commands

| Command | Effect |
|---------|--------|
| `make install` | Install dependencies via Poetry |
| `make run` | Run the main pipeline |
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
| **pylint** | Static analysis with a minimum score of 9.0 |
| **black** | Formatting check |
| **build** | Dependency installation smoke test |
| **test** | Unit tests with coverage reporting |

### Logging

All diagnostic output must use the shared Logger singleton from `src/util/logger.py`. Never use `print()`.

```python
from src.util import logger

logger.info("Training started for %s", ticker)
logger.warning("R2 is below threshold: %.4f", r2)
logger.error("Failed to load data for %s", ticker)
logger.debug("Feature matrix shape: %s", X.shape)
```
