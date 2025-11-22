clean:
	@rm -rf plots/*
	@rm -rf models/*
	@rm -rf invalid_models/*
	@rm -rf results/*
	@rm -rf params/*

run:
	@poetry run python3 src/main.py

install:
	@poetry lock
	@poetry install

ctrain:
	@make clean
	@make run

.PHONY: clean run install
