# Hyperion — Agent Guide

## Project Overview

Hyperion is a stock trading prediction system that trains machine-learning models on historical price data and uses them to simulate or execute trades. The target return on the portfolio is 5–10% ROI.

The main sub-project is:

| Sub-project | Language | Status | Purpose |
|---|---|---|---|
| `hyperion-py` | Python 3.12 | Active | XGBoost / LightGBM model training, prediction, and trading simulation |

---

## Repository Layout

```
hyperion/
├── src/
│   ├── align/        # Alignment / data-alignment logic
│   ├── data/         # Data fetching and preprocessing
│   ├── experimental/ # Walk-forward validation (experimental)
│   ├── feature/      # Feature engineering
│   ├── model/        # Model training and evaluation
│   ├── optimise/     # Hyperparameter optimisation (Optuna)
│   ├── pipeline/     # End-to-end training/prediction pipelines
│   ├── simulation/   # Trading simulation
│   ├── util/         # Shared utilities (logger singleton)
│   ├── visualisation/# Plots and charts
│   ├── writer/       # Results / model persistence
│   ├── main.py       # CLI entry-point
│   └── train.py      # Training entry-point
├── resources/        # Static assets / ticker lists
├── pyproject.toml    # Poetry project definition
├── Makefile          # Developer commands (see below)
├── .pre-commit-config.yaml
└── AGENTS.md         # This file
```

---

## Python Sub-project

### Setup

```bash
make install   # poetry lock && poetry install
```

Python **3.12** is required. Dependencies are managed with [Poetry](https://python-poetry.org/).

### Running

```bash
make run       # poetry run python3 src/main.py
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
pre-commit install
```

### Logging

**Never use `print()` for diagnostic or informational output.** All output must go through the shared `Logger` singleton in `src/util/logger.py`.

Import and use it like this:

```python
from src.util import logger

logger.info("Training started for %s", ticker)
logger.warning("R² is below threshold: %.4f", r2)
logger.error("Failed to load data for %s", ticker)
logger.debug("Feature matrix shape: %s", X.shape)
```

| Method | When to use |
|---|---|
| `logger.info()` | Normal progress messages (pipeline steps, results summaries) |
| `logger.warning()` | Recoverable anomalies (low R², missing data, skipped tickers) |
| `logger.error()` | Unexpected failures that affect output |
| `logger.debug()` | Verbose internals useful for troubleshooting only |

The logger is a singleton — importing it from any module always returns the same instance, so no additional configuration is needed.

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
2. Keep the alignment logic in `src/align/`.
3. Experimental features go under `src/experimental/` and must be clearly documented as such.

### Dependencies

- Python: add dependencies via `poetry add <package>` — never edit `pyproject.toml` by hand for dependencies.

### Testing

```bash
make test       # poetry run pytest tests/ -v --tb=short
make test-cov   # run tests with coverage report
```
