"""Dashboard configuration."""
import os

API_BASE_URL = os.getenv("GRIDPULSE_API_URL", "http://localhost:8000/api/v1")

# Datasets & models available (khớp với những gì bạn đã backfill)
DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]
MODELS = ["LSTM_h128_L2", "GRU_h128_L2", "VanillaRNN_h128_L2", "Seq2Seq_h128_L2"]

# Theme colors
COLOR_ACTUAL = "#1f2937"
COLOR_PREDICTED = "#ef4444"
COLOR_ANOMALY = "#f59e0b"
COLOR_ACCENT = "#3b82f6"