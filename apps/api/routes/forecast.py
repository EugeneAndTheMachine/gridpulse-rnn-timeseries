"""Forecast routes — query stored forecasts + on-demand inference."""
from datetime import datetime, timedelta, timezone

import torch
from fastapi import APIRouter, Depends, HTTPException, Query

from apps.api.dependencies import get_forecast_repo, get_models
from apps.api.schemas import (
    AccuracySummary,
    ForecastPoint,
    ForecastRequest,
    ForecastResponse,
)
from gridpulse.database.repositories import ForecastRepository
from gridpulse.serving.model_loader import ModelCache

router = APIRouter(prefix="/forecast")


@router.get("", response_model=ForecastResponse)
def get_forecasts(
    model_name: str | None = None,
    dataset: str | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    limit: int = Query(1000, ge=1, le=10000),
    repo: ForecastRepository = Depends(get_forecast_repo),
):
    """Query stored forecasts từ hypertable."""
    rows = repo.get_forecasts(
        model_name=model_name,
        dataset=dataset,
        start_time=start_time,
        end_time=end_time,
        limit=limit,
    )
    points = [
        ForecastPoint(
            time=r.time,
            horizon_step=r.horizon_step,
            predicted=r.predicted,
            actual=r.actual,
            residual=r.residual,
        )
        for r in rows
    ]
    return ForecastResponse(
        model_name=model_name or (rows[0].model_name if rows else ""),
        dataset=dataset or (rows[0].dataset if rows else ""),
        target_col=rows[0].target_col if rows else "OT",
        n_points=len(points),
        points=points,
    )


@router.post("/predict", response_model=ForecastResponse)
def predict(
    request: ForecastRequest,
    models: ModelCache = Depends(get_models),
):
    """On-demand inference với model đã trained (input: input_len × num_features)."""
    if not request.input_values:
        raise HTTPException(422, "input_values is empty")

    num_features = len(request.input_values[0])
    try:
        model = models.get(
            request.model_name,
            num_features=num_features,
            forecast_horizon=request.forecast_horizon,
        )
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(404, str(e))

    device = next(model.parameters()).device
    x = torch.tensor([request.input_values], dtype=torch.float32, device=device)
    with torch.no_grad():
        preds = model(x).cpu().numpy()[0]  # (horizon,)

    base = datetime.now(timezone.utc)
    points = [
        ForecastPoint(
            time=base + timedelta(hours=step),
            horizon_step=step + 1,
            predicted=float(p),
        )
        for step, p in enumerate(preds)
    ]
    return ForecastResponse(
        model_name=request.model_name,
        dataset="on-demand",
        target_col="OT",
        n_points=len(points),
        points=points,
    )


@router.get("/accuracy", response_model=AccuracySummary)
def accuracy(
    model_name: str,
    days: int = Query(7, ge=1, le=365),
    repo: ForecastRepository = Depends(get_forecast_repo),
):
    """Accuracy summary từ continuous aggregate forecast_accuracy_hourly."""
    summary = repo.get_accuracy_summary(model_name, days=days)
    return AccuracySummary(
        model_name=model_name,
        period_days=days,
        n_predictions=summary["n_predictions"],
        mae=summary["mae"],
        rmse=summary["rmse"],
    )
