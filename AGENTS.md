# Hyperion — Agent Guide

## Project Overview

Hyperion is a stock trading prediction system that trains machine-learning models on historical price data and uses them to simulate or execute trades. The target return on the portfolio is 5–10% ROI.

The main sub-project is:

| Sub-project | Language | Status | Purpose |
|---|---|---|---|
| `hyperion-py` | Python 3.12 | Active | XGBoost / LightGBM model training, prediction, trading simulation, and a Flask HTTP server |

---

## Repository Layout

```
hyperion/
├── hyperion-py/          # Python sub-project
│   ├── src/
│   │   ├── align/        # Alignment / data-alignment logic
│   │   ├── data/         # Data fetching and preprocessing
│   │   ├── experimental/ # Walk-forward validation (experimental)
│   │   ├── feature/      # Feature engineering
│   │   ├── model/        # Model training and evaluation
│   │   ├── optimise/     # Hyperparameter optimisation (Optuna)
│   │   ├── pipeline/     # End-to-end training/prediction pipelines
│   │   ├── server/       # Flask HTTP server (train / predict endpoints)
│   │   ├── simulation/   # Trading simulation
│   │   ├── visualisation/# Plots and charts
│   │   ├── writer/       # Results / model persistence
│   │   ├── main.py       # CLI entry-point
│   │   └── train.py      # Training entry-point
│   ├── resources/        # Static assets / ticker lists
│   ├── pyproject.toml    # Poetry project definition
│   ├── Makefile          # Developer commands (see below)
│   └── .pre-commit-config.yaml
│
└── AGENTS.md             # This file
```

---

## Python Sub-project (`hyperion-py`)

### Setup

```bash
cd hyperion-py
make install   # poetry lock && poetry install
```

Python **3.12** is required. Dependencies are managed with [Poetry](https://python-poetry.org/).

### Running

```bash
make run       # poetry run python3 src/main.py
```

### Cleaning

| Command | Effect |
|---|---|
| `make clean` | Remove plots, invalid_models, results, params |
| `make cleanmodels` | Remove plots, models, invalid_models, results, params |
| `make ctrain` | `clean` then `run` |
| `make cmtrain` | `cleanmodels` then `run` |

### HTTP Server Endpoints

The `server` package exposes a Flask server on port **8080**:

| Method | Path | Description |
|---|---|---|
| POST | `/train` | Train a model for the given ticker |
| GET | `/predict/<ticker>` | Predict the next 180 days for a ticker |
| GET | `/trading_results/<ticker>` | Fetch trading simulation results |

Example request body for `/train`:
```json
{ "ticker": "AAPL", "period": "5y", "interval": "1d" }
```

### Code Style

- **Formatter:** [Black](https://black.readthedocs.io/) — line length **120**.
- **Linter:** [Pylint](https://pylint.org/) — max line length 120; several noisy checks disabled (see `.pylintrc`).
- **Pre-commit hooks** (run automatically on commit and push):
  - `poetry-lock`, `poetry-check`, `poetry-install`
  - Syntax checks: `check-ast`, `check-json`, `check-toml`, `check-yaml`
  - Style: `end-of-file-fixer`, `trailing-whitespace`, `black`

Install hooks once after cloning:
```bash
cd hyperion-py
pre-commit install
```

---

## General Conventions

### Git & Pull Requests

- Branch off `main` for every change.
- PR titles and descriptions should follow the repository template (`.github/pull_request_template.md`):
  - **Description** — what the PR does and why.
  - **Changes in this pull request** — a bullet-point list of individual changes.
- GitHub Actions label PRs automatically via `.github/labeler.yml`.
- Keep commits atomic and descriptive.

### Adding Features

1. Implement new features under the appropriate `src/` sub-package, or create a new sub-package if none fits.
2. Keep the alignment logic in `hyperion-py/src/align/`.
3. Experimental features go under `hyperion-py/src/experimental/` and must be clearly documented as such.

### Dependencies

- Python: add dependencies via `poetry add <package>` — never edit `pyproject.toml` by hand for dependencies.

### Testing

There are currently no automated test suites. Validate changes manually using the Makefile commands above.
