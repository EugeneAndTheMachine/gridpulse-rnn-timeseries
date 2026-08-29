"""Forecasting page — historical predictions vs actual."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve()
while _ROOT != _ROOT.parent and not (_ROOT / "pyproject.toml").exists():
    _ROOT = _ROOT.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from apps.dashboard.components import api_client
from apps.dashboard.components.api_client import APIError
from apps.dashboard.components.charts import forecast_line_chart, residual_chart
from apps.dashboard.config import DATASETS, MODELS

st.set_page_config(page_title="Forecasting · GridPulse", page_icon="📈", layout="wide")
st.title("📈 Forecasting")

# --- Controls ---
c1, c2, c3 = st.columns([2, 2, 1])
with c1:
    model = st.selectbox("Model", MODELS)
with c2:
    dataset = st.selectbox("Dataset", DATASETS)
with c3:
    limit = st.number_input("Max points", 100, 10000, 2000, step=100)

# --- Fetch ---
try:
    with st.spinner("Loading forecasts..."):
        df = api_client.get_forecast_history(model, dataset, limit=int(limit))
except APIError as e:
    st.error(f"Failed to load: {e}")
    st.stop()

if df.empty:
    st.warning(f"No forecasts found for {model} on {dataset}. Did you run the backfill script?")
    st.code(f"python scripts/backfill_forecasts.py --model {model} --dataset {dataset}")
    st.stop()

# --- Optionally filter to horizon_step == 1 for a clean single line ---
if "horizon_step" in df.columns:
    step = st.slider("Horizon step to display", 1, int(df["horizon_step"].max()), 1)
    df_plot = df[df["horizon_step"] == step].copy()
else:
    df_plot = df

# --- Metrics row ---
acc = None
try:
    acc = api_client.get_accuracy(model, days=30)
except APIError:
    pass

m1, m2, m3, m4 = st.columns(4)
m1.metric("Points shown", len(df_plot))
if acc and acc.get("mae") is not None:
    m2.metric("MAE (30d)", f"{acc['mae']:.4f}")
    m3.metric("RMSE (30d)", f"{acc['rmse']:.4f}")
    m4.metric("Predictions (30d)", f"{acc['n_predictions']:,}")

# --- Charts ---
st.plotly_chart(forecast_line_chart(df_plot, f"{model} · {dataset}"), use_container_width=True)

if "residual" in df_plot.columns and df_plot["residual"].notna().any():
    st.plotly_chart(residual_chart(df_plot), use_container_width=True)

# --- Raw data expander ---
with st.expander("View raw data"):
    st.dataframe(df_plot, use_container_width=True, hide_index=True)
    st.download_button(
        "Download CSV",
        df_plot.to_csv(index=False).encode(),
        file_name=f"{model}_{dataset}_forecasts.csv",
        mime="text/csv",
    )