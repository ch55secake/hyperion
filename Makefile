## Configurable parameters (override via ARGS or per-parameter variables)
## Examples:
##   make run ARGS="--period 5y --n-trials 200"
##   make run PERIOD=5y N_TRIALS=200
##   make run INITIAL_CAPITAL=50000 TEST_SIZE=0.3
PERIOD          ?= 2y
INTERVALS       ?= 1d,1h
TEST_SIZE       ?= 0.2
TARGET_DAYS     ?= 10
N_TRIALS        ?= 1000
R2_SAVE         ?= 0.0012
R2_INVALID      ?= -0.3
INITIAL_CAPITAL ?= 10000
TRANSACTION_COST ?= 0.001
ARGS            ?=

_RUN_ARGS = --period $(PERIOD) \
            --intervals $(INTERVALS) \
            --test-size $(TEST_SIZE) \
            --target-days $(TARGET_DAYS) \
            --n-trials $(N_TRIALS) \
            --r2-save-threshold $(R2_SAVE) \
            --r2-invalid-threshold $(R2_INVALID) \
            --initial-capital $(INITIAL_CAPITAL) \
            --transaction-cost $(TRANSACTION_COST) \
            $(ARGS)

cleanmodels:
	@rm -rf plots/*
	@rm -rf models/*
	@rm -rf invalid_models/*
	@rm -rf results/*
	@rm -rf params/*

clean:
	@rm -rf plots/*
	@rm -rf invalid_models/*
	@rm -rf results/*
	@rm -rf params/*

run:
	@uv run python3 src/main.py $(_RUN_ARGS)

install:
	@uv sync

cmtrain:
	@make cleanmodels
	@make run

ctrain:
	@make clean
	@make run


test:
	@uv run pytest tests/ -v --tb=short

test-cov:
	@uv run pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

help:
	@uv run python3 src/main.py --help

.PHONY: clean cleanmodels run install cmtrain ctrain test test-cov help
