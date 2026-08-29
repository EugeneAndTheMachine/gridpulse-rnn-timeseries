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

db-migrate:
	uv run alembic upgrade head

db-revision:
	uv run alembic revision --autogenerate -m "$(msg)"

db-shell:
	docker compose exec db psql -U gridpulse -d gridpulse

db-reset:
	docker compose down -v
	docker compose up -d db
	sleep 5
	uv run alembic stamp head

dashboard-dev:
	uv run streamlit run apps/dashboard/Home.py

dashboard-run:
	uv run streamlit run apps/dashboard/Home.py --server.port 8501 --server.address 0.0.0.0

seed:
	bash scripts/seed_all.sh

demo: docker-up seed
	@echo "Stack ready → API :8000/docs · Dashboard :8501"
