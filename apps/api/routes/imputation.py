"""Imputation endpoints."""
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/imputation")


class ImputeRequest(BaseModel):
    values: list[float | None] = Field(..., description="Series with nulls as missing")
    method: str = Field("linear", description="linear | forward_fill | mean")


class ImputeResponse(BaseModel):
    method: str
    n_missing: int
    imputed: list[float]


@router.post("/impute", response_model=ImputeResponse)
def impute_series(req: ImputeRequest):
    import pandas as pd
    s = pd.Series([np.nan if v is None else v for v in req.values])
    n_missing = int(s.isna().sum())

    if req.method == "linear":
        out = s.interpolate(method="linear", limit_direction="both")
    elif req.method == "forward_fill":
        out = s.ffill().bfill()
    elif req.method == "mean":
        out = s.fillna(s.mean())
    else:
        raise HTTPException(400, f"Unknown method: {req.method}")

    return ImputeResponse(
        method=req.method,
        n_missing=n_missing,
        imputed=out.tolist(),
    )