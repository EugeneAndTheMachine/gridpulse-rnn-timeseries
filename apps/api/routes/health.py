from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from apps.api.config import settings
from apps.api.dependencies import get_db_session, get_models
from apps.api.schemas import HealthResponse
from gridpulse.serving.model_loader import ModelCache

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(
    session: Session = Depends(get_db_session),
    models: ModelCache = Depends(get_models),
):
    # Test DB
    try:
        session.execute(text("SELECT 1"))
        db_status = "ok"
    except Exception:
        db_status = "error"

    return HealthResponse(
        status="ok" if db_status == "ok" else "degraded",
        api_version=settings.api_version,
        database=db_status,
        models_loaded=models.size,
    )