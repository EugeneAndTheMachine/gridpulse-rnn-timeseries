"""Repository pattern: abstract DB access from business logic."""
from __future__ import annotations
from datetime import datetime
from typing import Any, Iterable

import pandas as pd
from sqlalchemy import select, and_, desc, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from gridpulse.database.models import Forecast, AnomalyEvent, ModelRegistry
from gridpulse.utils.logger import logger


class ForecastRepository:
    """CRUD + query operations cho forecasts hypertable."""

    def __init__(self, session: Session):
        self.session = session

    def bulk_insert(
        self,
        records: Iterable[dict[str, Any]],
        on_conflict: str = "update",
    ) -> int:
        """
        Bulk insert forecasts. Handle duplicates với UPSERT.

        records: list of dicts với keys matching Forecast columns
        on_conflict: 'update' (upsert) or 'ignore' (skip duplicates)
        """
        records_list = list(records)
        if not records_list:
            return 0

        stmt = pg_insert(Forecast).values(records_list)

        if on_conflict == "update":
            stmt = stmt.on_conflict_do_update(
                index_elements=["time", "model_name", "horizon_step"],
                set_={
                    "predicted": stmt.excluded.predicted,
                    "actual": stmt.excluded.actual,
                },
            )
        elif on_conflict == "ignore":
            stmt = stmt.on_conflict_do_nothing(
                index_elements=["time", "model_name", "horizon_step"],
            )

        result = self.session.execute(stmt)
        self.session.flush()
        logger.info(f"Inserted {len(records_list)} forecasts")
        return result.rowcount

    def get_forecasts(
        self,
        model_name: str | None = None,
        dataset: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 1000,
    ) -> list[Forecast]:
        """Query forecasts với filters."""
        stmt = select(Forecast)
        conditions = []
        if model_name:
            conditions.append(Forecast.model_name == model_name)
        if dataset:
            conditions.append(Forecast.dataset == dataset)
        if start_time:
            conditions.append(Forecast.time >= start_time)
        if end_time:
            conditions.append(Forecast.time <= end_time)

        if conditions:
            stmt = stmt.where(and_(*conditions))

        stmt = stmt.order_by(desc(Forecast.time)).limit(limit)
        return list(self.session.scalars(stmt))

    def get_forecasts_as_dataframe(
        self,
        model_name: str | None = None,
        dataset: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> pd.DataFrame:
        """Return forecasts as pandas DataFrame — tiện cho analysis."""
        forecasts = self.get_forecasts(model_name, dataset, start_time, end_time, limit=100000)
        if not forecasts:
            return pd.DataFrame()

        return pd.DataFrame([
            {
                "time": f.time,
                "model_name": f.model_name,
                "dataset": f.dataset,
                "target_col": f.target_col,
                "horizon_step": f.horizon_step,
                "predicted": f.predicted,
                "actual": f.actual,
                "residual": f.residual,
            }
            for f in forecasts
        ])

    def get_accuracy_summary(
        self, model_name: str, days: int = 7
    ) -> dict[str, float]:
        """
        Fast accuracy query using the continuous aggregate.
        Not the raw hypertable — much faster.
        """
        sql = text("""
            SELECT
                AVG(mae) AS mae,
                AVG(rmse) AS rmse,
                SUM(n_predictions) AS n_predictions
            FROM forecast_accuracy_hourly
            WHERE model_name = :model
              AND bucket >= NOW() - make_interval(days => :days)
        """)
        result = self.session.execute(
            sql, {"model": model_name, "days": days}
        ).mappings().one_or_none()

        if result is None or result["n_predictions"] is None:
            return {"mae": None, "rmse": None, "n_predictions": 0}

        return {
            "mae": float(result["mae"]) if result["mae"] else None,
            "rmse": float(result["rmse"]) if result["rmse"] else None,
            "n_predictions": int(result["n_predictions"]),
        }

    def get_metrics(
        self,
        model_name: str,
        dataset: str | None = None,
        horizon_step: int | None = None,
    ) -> dict[str, Any]:
        """
        Accuracy computed directly from the forecasts table over the full
        available range (works with historical data, unlike the NOW()-relative
        continuous aggregate). residual is stored as (predicted - actual).
        """
        sql = text("""
            SELECT
                count(*) FILTER (WHERE actual IS NOT NULL)              AS n,
                AVG(abs(residual))                                     AS mae,
                sqrt(AVG(residual * residual))                        AS rmse,
                AVG(2 * abs(residual)
                    / NULLIF(abs(actual) + abs(predicted), 0)) * 100  AS smape
            FROM forecasts
            WHERE model_name = :model
              AND actual IS NOT NULL
              AND (:dataset IS NULL OR dataset = :dataset)
              AND (:horizon IS NULL OR horizon_step = :horizon)
        """)
        row = self.session.execute(
            sql,
            {"model": model_name, "dataset": dataset, "horizon": horizon_step},
        ).mappings().one_or_none()

        if row is None or not row["n"]:
            return {"n_predictions": 0, "mae": None, "rmse": None, "smape": None}
        return {
            "n_predictions": int(row["n"]),
            "mae": float(row["mae"]) if row["mae"] is not None else None,
            "rmse": float(row["rmse"]) if row["rmse"] is not None else None,
            "smape": float(row["smape"]) if row["smape"] is not None else None,
        }


class AnomalyRepository:
    """Anomaly events CRUD."""

    def __init__(self, session: Session):
        self.session = session

    def create(self, **kwargs) -> AnomalyEvent:
        event = AnomalyEvent(**kwargs)
        self.session.add(event)
        self.session.flush()
        return event

    def bulk_create(self, events: Iterable[dict[str, Any]]) -> int:
        events_list = list(events)
        if not events_list:
            return 0
        self.session.execute(pg_insert(AnomalyEvent).values(events_list))
        self.session.flush()
        return len(events_list)

    def list_recent(
        self, model_name: str | None = None, limit: int = 50
    ) -> list[AnomalyEvent]:
        stmt = select(AnomalyEvent).order_by(desc(AnomalyEvent.start_time)).limit(limit)
        if model_name:
            stmt = stmt.where(AnomalyEvent.model_name == model_name)
        return list(self.session.scalars(stmt))

    def confirm(self, event_id: int, notes: str | None = None) -> AnomalyEvent | None:
        event = self.session.get(AnomalyEvent, event_id)
        if event:
            event.is_confirmed = True
            if notes:
                event.notes = notes
        return event


class ModelRegistryRepository:
    """Model metadata CRUD."""

    def __init__(self, session: Session):
        self.session = session

    def register(
        self,
        model_name: str,
        version: str,
        task: str,
        architecture: str,
        checkpoint_path: str,
        **kwargs,
    ) -> ModelRegistry:
        model = ModelRegistry(
            model_name=model_name,
            version=version,
            task=task,
            architecture=architecture,
            checkpoint_path=checkpoint_path,
            **kwargs,
        )
        self.session.add(model)
        self.session.flush()
        logger.info(f"Registered model: {model_name} v{version}")
        return model

    def get_active_by_task(self, task: str) -> list[ModelRegistry]:
        stmt = select(ModelRegistry).where(
            and_(ModelRegistry.task == task, ModelRegistry.is_active.is_(True))
        )
        return list(self.session.scalars(stmt))

    def get(self, model_name: str, version: str) -> ModelRegistry | None:
        stmt = select(ModelRegistry).where(
            and_(
                ModelRegistry.model_name == model_name,
                ModelRegistry.version == version,
            )
        )
        return self.session.scalars(stmt).one_or_none()

    def deactivate(self, model_name: str, version: str) -> None:
        model = self.get(model_name, version)
        if model:
            model.is_active = False