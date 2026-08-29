"""GridPulse Dashboard — Home."""
import sys
from pathlib import Path

# Ensure the project root is importable so `apps.*` resolves under `streamlit run`
# (Streamlit puts the script's own folder on sys.path, not the project root).
_ROOT = Path(__file__).resolve()
while _ROOT != _ROOT.parent and not (_ROOT / "pyproject.toml").exists():
    _ROOT = _ROOT.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from apps.dashboard.components import api_client
from apps.dashboard.components.api_client import APIError

st.set_page_config(
    page_title="GridPulse",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("⚡ GridPulse")
st.caption("Time-Series Forecasting · Imputation · Anomaly Detection — powered by RNN/LSTM/GRU")

# --- API health banner ---
try:
    h = api_client.health()
    if h["status"] == "ok":
        st.success(f"API connected · DB: {h['database']} · Models loaded: {h['models_loaded']}")
    else:
        st.warning(f"API degraded · DB: {h.get('database')}")
except APIError as e:
    st.error(f"Cannot connect to API. Is it running? Details: {e}")
    st.info("Start it with `make docker-up` or `uv run uvicorn apps.api.main:app`")
    st.stop()

st.divider()

# --- Project overview ---
col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("📈 Forecasting")
    st.write("Multi-horizon predictions from RNN-family models, compared against classical baselines.")
with col2:
    st.subheader("🔧 Imputation")
    st.write("Six methods benchmarked on natural + synthetic missingness with mask-and-recover.")
with col3:
    st.subheader("🚨 Anomaly Detection")
    st.write("Residual-based detection with dynamic thresholding and event grouping.")

st.divider()

# --- Quick stats ---
st.subheader("Registered Models")
try:
    models = api_client.list_models()
    if models:
        import pandas as pd
        df = pd.DataFrame(models)
        cols = [c for c in ["model_name", "architecture", "task", "forecast_horizon", "num_parameters"] if c in df.columns]
        st.dataframe(df[cols], use_container_width=True, hide_index=True)
    else:
        st.info("No models registered yet. Run `scripts/backfill_forecasts.py`.")
except APIError as e:
    st.warning(f"Could not load models: {e}")

st.caption("Use the sidebar to navigate between pages →")