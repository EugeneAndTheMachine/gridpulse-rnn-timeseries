"""Model Comparison page — accuracy across models."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve()
while _ROOT != _ROOT.parent and not (_ROOT / "pyproject.toml").exists():
    _ROOT = _ROOT.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st

from apps.dashboard.components import api_client
from apps.dashboard.components.api_client import APIError
from apps.dashboard.config import MODELS

st.set_page_config(page_title="Model Comparison · GridPulse", page_icon="📊", layout="wide")
st.title("📊 Model Comparison")

days = st.slider("Accuracy window (days)", 1, 90, 30)

rows = []
for model in MODELS:
    try:
        acc = api_client.get_accuracy(model, days=days)
        if acc.get("mae") is not None:
            rows.append({
                "model": model,
                "mae": acc["mae"],
                "rmse": acc["rmse"],
                "n_predictions": acc["n_predictions"],
            })
    except APIError:
        continue

if not rows:
    st.warning("No accuracy data. Backfill forecasts for multiple models first.")
    st.stop()

df = pd.DataFrame(rows).sort_values("mae")

# --- Metric bars ---
c1, c2 = st.columns(2)
with c1:
    fig_mae = px.bar(df, x="model", y="mae", title="MAE by Model (lower is better)",
                     color="mae", color_continuous_scale="RdYlGn_r")
    st.plotly_chart(fig_mae, use_container_width=True)
with c2:
    fig_rmse = px.bar(df, x="model", y="rmse", title="RMSE by Model",
                      color="rmse", color_continuous_scale="RdYlGn_r")
    st.plotly_chart(fig_rmse, use_container_width=True)

# --- Best model callout ---
best = df.iloc[0]
st.success(f"🏆 Best model by MAE: **{best['model']}** (MAE={best['mae']:.4f}, RMSE={best['rmse']:.4f})")

# --- Table ---
st.dataframe(df, use_container_width=True, hide_index=True)

# --- Static benchmark table (from your notebooks) for context ---
with st.expander("📚 Research benchmark (from notebooks, ETTh1)"):
    st.markdown("""
    | Horizon | Best Model | MAE | RMSE | sMAPE |
    |---|---|---|---|---|
    | 1h | Persistence | 0.130 | 0.191 | 28.76% |
    | 6h | Linear Ridge | 0.227 | 0.324 | 44.67% |
    | 24h | Linear Ridge | 0.386 | 0.519 | 68.60% |
    | 48h | Linear Ridge | 0.493 | 0.647 | 83.39% |
    """)