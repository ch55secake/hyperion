# Hyperion

> [!NOTE]
> This project is under active development.

## Overview

Hyperion is a stock trading prediction system that trains machine-learning models on historical price data and uses them to simulate trades. The target return on the portfolio is **5–10% ROI**.

## Features

- XGBoost and LightGBM model training on historical stock data
- Stacked multi-timeframe model (daily + hourly, experimental)
- Hyperparameter optimisation via Optuna
- Prediction analysis and data visualisation
- Walk-forward validation (experimental)
- Trading simulation

## Getting Started

Python **3.12** and [Poetry](https://python-poetry.org/) are required.

```bash
cd hyperion-py
make install   # poetry lock && poetry install
make run       # poetry run python3 src/main.py
```

## HTTP Server

A Flask server is exposed on port **8080** with the following endpoints:

| Method | Path | Description |
|---|---|---|
| POST | `/train` | Train a model for the given ticker |
| GET | `/predict/<ticker>` | Predict the next 180 days for a ticker |
| GET | `/trading_results/<ticker>` | Fetch trading simulation results |

### Example requests

**Train:**
```bash
curl -X POST http://localhost:8080/train \
  -H 'Content-Type: application/json' \
  -d '{"ticker": "AAPL", "period": "5y", "interval": "1d"}'
```

**Predict:**
```bash
curl http://localhost:8080/predict/AAPL
```

**Trading results:**
```bash
curl http://localhost:8080/trading_results/AAPL
```

## Makefile Commands

| Command | Effect |
|---|---|
| `make install` | Install dependencies via Poetry |
| `make run` | Run the application |
| `make clean` | Remove plots, invalid models, results and params |
| `make cleanmodels` | Remove plots, models, invalid models, results and params |
| `make ctrain` | `clean` then `run` |
| `make cmtrain` | `cleanmodels` then `run` |