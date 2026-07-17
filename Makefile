.PHONY: setup test lint demo run walk-forward train-live predict-live forecast dashboard

setup:
	python3 -m venv .venv
	.venv/bin/python -m pip install -e ".[dev]"

test:
	.venv/bin/python -m pytest

lint:
	.venv/bin/ruff check src tests
	.venv/bin/ruff format --check src tests

demo:
	.venv/bin/market-regime demo --days 1200 --iterations 12

run:
	.venv/bin/market-regime run --config configs/default.yaml

walk-forward:
	.venv/bin/market-regime walk-forward --config configs/default.yaml

train-live:
	.venv/bin/market-regime train-live --config configs/default.yaml

predict-live:
	.venv/bin/market-regime predict-live --config configs/default.yaml

forecast:
	.venv/bin/market-regime forecast --config configs/default.yaml

dashboard:
	.venv/bin/streamlit run streamlit_app.py
