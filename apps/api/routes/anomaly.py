from fastapi import APIRouter, Depends, HTTPException, Query

from apps.api.dependencies import get_anomaly_repo
from apps.api.schemas import AnomalyEventResponse
from gridpulse.database.repositories import AnomalyRepository

router = APIRouter(prefix="/anomaly")


@router.get("/events", response_model=list[AnomalyEventResponse])
def list_events(
    model_name: str | None = None,
    limit: int = Query(50, ge=1, le=500),
    repo: AnomalyRepository = Depends(get_anomaly_repo),
):
    events = repo.list_recent(model_name=model_name, limit=limit)
    return events


@router.post("/events/{event_id}/confirm")
def confirm_event(
    event_id: int,
    notes: str | None = None,
    repo: AnomalyRepository = Depends(get_anomaly_repo),
):
    event = repo.confirm(event_id, notes)
    if event is None:
        raise HTTPException(404, f"Event {event_id} not found")
    return {"id": event_id, "confirmed": True}