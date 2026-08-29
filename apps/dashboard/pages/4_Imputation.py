"""Imputation demo page."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import httpx

from apps.dashboard.config import API_BASE_URL, COLOR_ACTUAL, COLOR_PREDICTED

st.set_page_config(page_title="Imputation · GridPulse", page_icon="🔧", layout="wide")
st.title("🔧 Missing Data Imputation")

st.write("Demonstrate imputation methods on a series with injected gaps.")

# --- Generate a demo series ---
c1, c2, c3 = st.columns(3)
with c1:
    n_points = st.slider("Series length", 50, 500, 200)
with c2:
    missing_pct = st.slider("Missing %", 5, 50, 20)
with c3:
    method = st.selectbox("Method", ["linear", "forward_fill", "mean"])

# Synthetic sine + noise
rng = np.random.RandomState(42)
t = np.arange(n_points)
clean = 10 + 5 * np.sin(2 * np.pi * t / 24) + rng.randn(n_points)

# Inject missing
corrupted = clean.copy()
missing_idx = rng.choice(n_points, size=int(n_points * missing_pct / 100), replace=False)
corrupted[missing_idx] = np.nan

# --- Call API ---
try:
    r = httpx.post(f"{API_BASE_URL}/imputation/impute", json={
        "values": [None if np.isnan(v) else float(v) for v in corrupted],
        "method": method,
    }, timeout=30)
    r.raise_for_status()
    result = r.json()
    imputed = np.array(result["imputed"])
except Exception as e:
    st.error(f"API call failed: {e}")
    st.stop()

# --- Metrics: error at masked positions only ---
mae = np.mean(np.abs(clean[missing_idx] - imputed[missing_idx]))
m1, m2 = st.columns(2)
m1.metric("Missing points", result["n_missing"])
m2.metric("MAE at masked positions", f"{mae:.4f}")

# --- Chart ---
fig = go.Figure()
fig.add_trace(go.Scatter(x=t, y=clean, name="Ground truth", line=dict(color=COLOR_ACTUAL, width=1)))
fig.add_trace(go.Scatter(x=t, y=imputed, name=f"Imputed ({method})",
                         line=dict(color=COLOR_PREDICTED, width=2, dash="dot")))
fig.add_trace(go.Scatter(x=missing_idx, y=imputed[missing_idx], name="Filled points",
                         mode="markers", marker=dict(color="orange", size=6)))
fig.update_layout(title="Imputation Result", height=450, hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

# --- Reference: benchmark from notebook ---
with st.expander("📚 Imputation benchmark (from research notebook)"):
    st.markdown("""
    MAE by method on ETTh1 (mask-and-recover, lower is better):

    | Method | MCAR 10% | Block 24h |
    |---|---|---|
    | LinearInterp | 0.731 | 1.430 |
    | Forward Fill | 1.210 | 2.091 |
    | BiLSTM | 0.752 | 6.860 |
    | MICE | 2.218 | 8.845 |

    LinearInterp wins overall, especially for block gaps.
    """)