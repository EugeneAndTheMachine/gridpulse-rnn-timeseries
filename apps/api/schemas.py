"""Pydantic schemas cho request/response."""
from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field, ConfigDict


class HealthResponse(BaseModel):
    status: str
    api_version: str
    database: str
    models_loaded: int


class ForecastPoint(BaseModel):
    time: datetime
    horizon_step: int
    predicted: float
    actual: float | None = None
    residual: float | None = None


class ForecastResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    model_name: str
    dataset: str
    target_col: str
    n_points: int
    points: list[ForecastPoint]


class ForecastQuery(BaseModel):
    model_name: str | None = None
    dataset: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    limit: int = Field(default=1000, ge=1, le=10000)


class ForecastRequest(BaseModel):
    """On-demand forecast request — inference với model đã load."""
    model_name: str = Field(default="LSTM_h128_L2")
    input_values: list[list[float]] = Field(
        ..., description="Shape: (input_len, num_features). Ex: 96 timesteps × 30 features"
    )
    forecast_horizon: int = Field(default=24, ge=1, le=96)


class AnomalyEventResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    start_time: datetime
    end_time: datetime
    duration: int
    model_name: str
    dataset: str
    target_col: str
    peak_score: float
    mean_score: float
    threshold_type: str
    is_confirmed: bool
    notes: str | None


class ModelMetadata(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    model_name: str
    version: str
    task: str
    architecture: str
    forecast_horizon: int | None
    num_parameters: int | None
    metrics: dict[str, Any] | None
    is_active: bool
    created_at: datetime


class AccuracySummary(BaseModel):
    model_name: str
    period_days: int
    n_predictions: int
    mae: float | None
    rmse: float | None