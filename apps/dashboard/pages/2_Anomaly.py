"""Anomaly Detection page."""
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
from apps.dashboard.components.charts import anomaly_timeline_chart
from apps.dashboard.config import MODELS

st.set_page_config(page_title="Anomalies · GridPulse", page_icon="🚨", layout="wide")
st.title("🚨 Anomaly Detection")

c1, c2 = st.columns([2, 1])
with c1:
    model = st.selectbox("Model", ["(all)"] + MODELS)
with c2:
    limit = st.number_input("Max events", 10, 500, 100, step=10)

model_filter = None if model == "(all)" else model

try:
    events = api_client.get_anomaly_events(model_filter, limit=int(limit))
except APIError as e:
    st.error(f"Failed to load events: {e}")
    st.stop()

if events.empty:
    st.info("No anomaly events found. Run the anomaly backfill script to populate them.")
    st.stop()

# --- Summary metrics ---
m1, m2, m3 = st.columns(3)
m1.metric("Total events", len(events))
if "is_confirmed" in events.columns:
    m2.metric("Confirmed", int(events["is_confirmed"].sum()))
if "peak_score" in events.columns:
    m3.metric("Max peak score", f"{events['peak_score'].max():.3f}")

# --- Timeline (needs the underlying series; fetch forecast history for context) ---
if model_filter:
    try:
        series = api_client.get_forecast_history(model_filter, limit=3000)
        if not series.empty and "horizon_step" in series.columns:
            series = series[series["horizon_step"] == 1]
        if not series.empty:
            st.plotly_chart(
                anomaly_timeline_chart(series, events, value_col="actual"),
                use_container_width=True,
            )
    except APIError:
        pass

# --- Events table with confirm action ---
st.subheader("Detected Events")
st.dataframe(
    events[[c for c in ["id", "start_time", "end_time", "duration", "model_name",
                        "peak_score", "threshold_type", "is_confirmed"] if c in events.columns]],
    use_container_width=True, hide_index=True,
)

# --- Confirm an event ---
with st.expander("Confirm an event (human review)"):
    event_id = st.number_input("Event ID", min_value=1, step=1)
    notes = st.text_input("Notes (optional)")
    if st.button("Mark as confirmed"):
        try:
            import httpx
            from apps.dashboard.config import API_BASE_URL
            r = httpx.post(f"{API_BASE_URL}/anomaly/events/{int(event_id)}/confirm",
                          params={"notes": notes} if notes else None, timeout=15)
            r.raise_for_status()
            st.success(f"Event {int(event_id)} confirmed.")
            st.cache_data.clear()
        except Exception as e:
            st.error(f"Failed: {e}")