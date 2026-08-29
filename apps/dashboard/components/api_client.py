"""HTTP client wrapping the GridPulse API."""
from __future__ import annotations
from datetime import datetime
from typing import Any

import httpx
import pandas as pd
import streamlit as st

from apps.dashboard.config import API_BASE_URL


class APIError(Exception):
    pass


def _get(path: str, params: dict | None = None) -> Any:
    url = f"{API_BASE_URL}{path}"
    try:
        resp = httpx.get(url, params=params, timeout=30.0)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        raise APIError(f"{e.response.status_code}: {e.response.text[:200]}")
    except httpx.RequestError as e:
        raise APIError(f"Cannot reach API at {url}: {e}")


def _post(path: str, json: dict) -> Any:
    url = f"{API_BASE_URL}{path}"
    try:
        resp = httpx.post(url, json=json, timeout=60.0)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        raise APIError(f"{e.response.status_code}: {e.response.text[:200]}")
    except httpx.RequestError as e:
        raise APIError(f"Cannot reach API at {url}: {e}")


# --- Cached wrappers (Streamlit caches to avoid hammering API) ---

@st.cache_data(ttl=60)
def health() -> dict:
    return _get("/health")


@st.cache_data(ttl=60)
def list_models(task: str | None = None) -> list[dict]:
    params = {"task": task} if task else None
    return _get("/models", params=params)


@st.cache_data(ttl=60)
def available_models() -> dict:
    return _get("/models/available")


@st.cache_data(ttl=30)
def get_forecast_history(
    model_name: str,
    dataset: str = "ETTh1",
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    limit: int = 2000,
) -> pd.DataFrame:
    """
    NOTE: adjust the path below to match your actual route.
    Status doc shows `GET /forecast`; tuần-12 draft used `/forecast/history`.
    """
    params: dict[str, Any] = {"model_name": model_name, "dataset": dataset, "limit": limit}
    if start_time:
        params["start_time"] = start_time.isoformat()
    if end_time:
        params["end_time"] = end_time.isoformat()

    data = _get("/forecast", params=params)  # ← đổi thành "/forecast/history" nếu route bạn tên vậy
    points = data.get("points", [])
    if not points:
        return pd.DataFrame()

    df = pd.DataFrame(points)
    df["time"] = pd.to_datetime(df["time"])
    return df.sort_values("time").reset_index(drop=True)


@st.cache_data(ttl=30)
def get_accuracy(model_name: str, days: int = 7) -> dict:
    return _get("/forecast/accuracy", params={"model_name": model_name, "days": days})


@st.cache_data(ttl=30)
def get_anomaly_events(model_name: str | None = None, limit: int = 100) -> pd.DataFrame:
    params: dict[str, Any] = {"limit": limit}
    if model_name:
        params["model_name"] = model_name
    data = _get("/anomaly/events", params=params)
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    for col in ("start_time", "end_time"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df


def predict_now(model_name: str, input_values: list[list[float]], horizon: int = 24) -> pd.DataFrame:
    data = _post("/forecast/predict", {
        "model_name": model_name,
        "input_values": input_values,
        "forecast_horizon": horizon,
    })
    df = pd.DataFrame(data.get("points", []))
    if not df.empty:
        df["time"] = pd.to_datetime(df["time"])
    return df