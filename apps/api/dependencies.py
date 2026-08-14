"""FastAPI dependencies (DI)."""
from typing import Iterator
from sqlalchemy.orm import Session
from fastapi import Depends

from gridpulse.database.connection import get_session_factory
from gridpulse.database.repositories import (
    ForecastRepository, AnomalyRepository, ModelRegistryRepository,
)
from gridpulse.serving.model_loader import ModelCache, get_model_cache


def get_db_session() -> Iterator[Session]:
    """DB session per request."""
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()


def get_forecast_repo(
    session: Session = Depends(get_db_session),
) -> ForecastRepository:
    return ForecastRepository(session)


def get_anomaly_repo(
    session: Session = Depends(get_db_session),
) -> AnomalyRepository:
    return AnomalyRepository(session)


def get_registry_repo(
    session: Session = Depends(get_db_session),
) -> ModelRegistryRepository:
    return ModelRegistryRepository(session)


def get_models() -> ModelCache:
    return get_model_cache()