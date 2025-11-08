clean:
	@rm -rf plots/*
	@rm -rf models/*
	@rm -rf historic_data/*
	@rm -rf results/*

run:
	@poetry run python3 src/main.py

install:
	@poetry lock
	@poetry install

.PHONY: clean run install