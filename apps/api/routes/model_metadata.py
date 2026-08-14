from fastapi import APIRouter, Depends, HTTPException

from apps.api.dependencies import get_registry_repo, get_models
from apps.api.schemas import ModelMetadata
from gridpulse.database.repositories import ModelRegistryRepository
from gridpulse.serving.model_loader import ModelCache

router = APIRouter(prefix="/models")


@router.get("", response_model=list[ModelMetadata])
def list_models(
    task: str | None = None,
    repo: ModelRegistryRepository = Depends(get_registry_repo),
):
    if task:
        return repo.get_active_by_task(task)
    from sqlalchemy import select
    from gridpulse.database.models import ModelRegistry
    return list(repo.session.scalars(select(ModelRegistry)))


@router.get("/available")
def list_available_checkpoints(
    models: ModelCache = Depends(get_models),
):
    """Models có checkpoint on disk (không nhất thiết đã register)."""
    return {"available": models.list_available(), "loaded": models.size}