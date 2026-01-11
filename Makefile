.PHONY: format lint test

format:
	python -m ruff format .

lint:
	python -m ruff check .
	python -m mypy scenarioops

test:
	python -m pytest
