"""SQLAlchemy ORM models mirror init.sql schema."""
from __future__ import annotations
from datetime import datetime
from typing import Any

from sqlalchemy import (
    String, Integer, BigInteger, Float, Boolean, DateTime,
    Text, UniqueConstraint, Index
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from gridpulse.database.connection import Base


class Forecast(Base):
    """Individual forecast prediction — 1 row per (time, model, horizon_step)."""
    __tablename__ = "forecasts"

    # Composite key: time + model + horizon_step
    time: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    model_name: Mapped[str] = mapped_column(String(200), primary_key=True)
    horizon_step: Mapped[int] = mapped_column(Integer, primary_key=True)

    dataset: Mapped[str] = mapped_column(String(100), nullable=False)
    target_col: Mapped[str] = mapped_column(String(100), nullable=False)
    predicted: Mapped[float] = mapped_column(Float, nullable=False)
    actual: Mapped[float | None] = mapped_column(Float, nullable=True)
    # `residual` is a generated column in Postgres — SQLAlchemy chỉ đọc
    residual: Mapped[float | None] = mapped_column(Float, nullable=True)
    run_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index("idx_forecasts_model_time", "model_name", "time"),
        Index("idx_forecasts_dataset_time", "dataset", "time"),
        {"extend_existing": True},
    )

    def __repr__(self) -> str:
        return (
            f"Forecast(time={self.time}, model={self.model_name}, "
            f"step={self.horizon_step}, pred={self.predicted:.3f})"
        )


class AnomalyEvent(Base):
    """Grouped anomaly event — from event_grouping module."""
    __tablename__ = "anomaly_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    start_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    end_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    duration: Mapped[int] = mapped_column(Integer, nullable=False)

    model_name: Mapped[str] = mapped_column(String(200), nullable=False)
    dataset: Mapped[str] = mapped_column(String(100), nullable=False)
    target_col: Mapped[str] = mapped_column(String(100), nullable=False)

    peak_score: Mapped[float] = mapped_column(Float, nullable=False)
    mean_score: Mapped[float] = mapped_column(Float, nullable=False)
    threshold_type: Mapped[str] = mapped_column(String(50), nullable=False)
    threshold_value: Mapped[float | None] = mapped_column(Float, nullable=True)

    is_confirmed: Mapped[bool] = mapped_column(Boolean, default=False)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class ModelRegistry(Base):
    """Registered model metadata."""
    __tablename__ = "model_registry"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String(200), nullable=False)
    version: Mapped[str] = mapped_column(String(50), nullable=False)
    task: Mapped[str] = mapped_column(String(50), nullable=False)
    architecture: Mapped[str] = mapped_column(String(100), nullable=False)

    input_len: Mapped[int | None] = mapped_column(Integer, nullable=True)
    forecast_horizon: Mapped[int | None] = mapped_column(Integer, nullable=True)
    num_features: Mapped[int | None] = mapped_column(Integer, nullable=True)
    num_parameters: Mapped[int | None] = mapped_column(BigInteger, nullable=True)

    metrics: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    hyperparameters: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    checkpoint_path: Mapped[str] = mapped_column(Text, nullable=False)
    mlflow_run_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint("model_name", "version", name="uq_model_version"),
    )