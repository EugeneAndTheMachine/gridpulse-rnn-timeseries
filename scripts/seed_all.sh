#!/usr/bin/env bash
# Populate DB with forecasts + anomalies for all models. Run after `docker compose up`.
set -e

echo "Waiting for API to be healthy..."
until curl -sf http://localhost:8000/api/v1/health > /dev/null; do
  sleep 2
done

echo "Backfilling forecasts..."
for model in LSTM_h128_L2 GRU_h128_L2 VanillaRNN_h128_L2; do
  uv run python scripts/backfill_forecasts.py --model "$model" --dataset ETTh1
done

echo "Backfilling anomalies..."
uv run python scripts/backfill_anomalies.py --model LSTM_h128_L2 --dataset ETTh1 || echo "anomaly backfill skipped"

echo "Refreshing continuous aggregate..."
docker compose exec -T db psql -U gridpulse -d gridpulse -c \
  "CALL refresh_continuous_aggregate('forecast_accuracy_hourly', NULL, NULL);"

echo "Done. Dashboard: http://localhost:8501"