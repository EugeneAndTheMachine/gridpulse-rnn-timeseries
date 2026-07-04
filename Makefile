PYTHONPATH ?= src
export PYTHONPATH

.PHONY: setup sync download-data test lint clean

setup:
	uv sync --all-extras
	uv run pre-commit install

sync:
	uv sync

download-data:
	uv run python -m gridpulse.data.download_ett
	uv run python -m gridpulse.data.download_uci

test:
	uv run pytest tests/ -v

lint:
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/

format:
	uv run ruff format src/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docker-up:
	docker compose up -d

docker-down:
	docker compose down

notebook:
	uv run jupyter lab notebooks/
