FROM python:3.12-slim

# uv (fastest option)
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy dependency manifests first for better layer caching
COPY pyproject.toml uv.lock ./
COPY src/ ./src/
COPY apps/ ./apps/

# Dashboard only needs the base deps (streamlit, httpx, pandas) — no torch/dl extra
RUN uv sync --frozen

# src-layout package `gridpulse` lives here but isn't pip-installed
ENV PYTHONPATH=/app/src

EXPOSE 8501

# `--no-sync` keeps the env built above intact (mirrors apps/api/Dockerfile)
CMD ["uv", "run", "--no-sync", "streamlit", "run", "apps/dashboard/Home.py", \
     "--server.port", "8501", "--server.address", "0.0.0.0", "--server.headless", "true"]
