FROM python:3.12-slim

# uv (fastest option)
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy source
COPY pyproject.toml uv.lock ./
COPY src/ ./src/
COPY apps/ ./apps/

# Install
RUN uv sync --frozen --extra api --extra dl

# Copy models (mount as volume in prod)
COPY models/ ./models/

# src-layout package `gridpulse` lives here but isn't pip-installed
ENV PYTHONPATH=/app/src

EXPOSE 8000

# `uv run` re-syncs the env to default deps by default, which drops the
# `dl` extra (torch). `--no-sync` keeps the env built above (api + dl) intact.
CMD ["uv", "run", "--no-sync", "uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000"]