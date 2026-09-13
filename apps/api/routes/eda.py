"""Exploratory Data Analysis endpoints — computed on demand with pandas."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/eda")

# Project root: apps/api/routes/eda.py -> parents[3]
_ROOT = Path(__file__).resolve().parents[3]
_DATA_DIR = _ROOT / "data" / "raw" / "ett"

DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]

# Domain knowledge for the ETT (Electricity Transformer Temperature) datasets.
COLUMN_META: dict[str, dict[str, str]] = {
    "HUFL": {"role": "feature", "description": "High UseFul Load"},
    "HULL": {"role": "feature", "description": "High UseLess Load"},
    "MUFL": {"role": "feature", "description": "Middle UseFul Load"},
    "MULL": {"role": "feature", "description": "Middle UseLess Load"},
    "LUFL": {"role": "feature", "description": "Low UseFul Load"},
    "LULL": {"role": "feature", "description": "Low UseLess Load"},
    "OT":   {"role": "target",  "description": "Oil Temperature (prediction target)"},
}

DATASET_BLURB = {
    "ETTh1": "Electricity Transformer Temperature — station 1, hourly resolution.",
    "ETTh2": "Electricity Transformer Temperature — station 2, hourly resolution.",
    "ETTm1": "Electricity Transformer Temperature — station 1, 15-minute resolution.",
    "ETTm2": "Electricity Transformer Temperature — station 2, 15-minute resolution.",
}

_DOW_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _load(dataset: str) -> pd.DataFrame:
    path = _DATA_DIR / f"{dataset}.csv"
    if not path.exists():
        raise HTTPException(404, f"Dataset file not found: {path.name}")
    df = pd.read_csv(path, parse_dates=["date"])
    return df


@lru_cache(maxsize=8)
def _compute(dataset: str) -> dict:
    df = _load(dataset)
    df = df.sort_values("date").reset_index(drop=True)
    numeric = [c for c in df.columns if c != "date"]
    target = "OT"

    # --- meta ---
    freq = "hourly" if dataset.startswith("ETTh") else "15-minute"
    meta = {
        "rows": int(len(df)),
        "cols": len(numeric),
        "start": df["date"].min().isoformat(),
        "end": df["date"].max().isoformat(),
        "freq": freq,
        "target": target,
        "blurb": DATASET_BLURB.get(dataset, ""),
    }

    # --- columns ---
    columns = [
        {
            "name": c,
            "role": COLUMN_META.get(c, {}).get("role", "feature"),
            "description": COLUMN_META.get(c, {}).get("description", c),
            "dtype": str(df[c].dtype),
        }
        for c in numeric
    ]

    # --- per-column summary + missingness ---
    summary, missing = [], []
    for c in numeric:
        s = df[c]
        n_missing = int(s.isna().sum())
        summary.append({
            "column": c,
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": float(s.min()),
            "p25": float(s.quantile(0.25)),
            "median": float(s.median()),
            "p75": float(s.quantile(0.75)),
            "max": float(s.max()),
            "missing": n_missing,
            "missing_pct": round(100 * n_missing / len(df), 3),
        })
        missing.append({
            "column": c,
            "missing": n_missing,
            "missing_pct": round(100 * n_missing / len(df), 3),
        })

    # --- histogram of the target ---
    ot = df[target].dropna().to_numpy()
    counts, edges = np.histogram(ot, bins=30)
    histogram = {
        "column": target,
        "bins": [round(float(e), 3) for e in edges],
        "counts": [int(x) for x in counts],
    }

    # --- correlation matrix ---
    corr = df[numeric].corr().round(3)
    correlation = {
        "columns": numeric,
        "matrix": [[float(v) for v in row] for row in corr.to_numpy()],
    }

    # --- seasonality of the target ---
    d = df[["date", target]].dropna().copy()
    d["hour"] = d["date"].dt.hour
    d["dow"] = d["date"].dt.dayofweek
    d["month"] = d["date"].dt.month
    hourly = [
        {"hour": int(h), "mean": float(g[target].mean()), "std": float(g[target].std())}
        for h, g in d.groupby("hour")
    ]
    weekly = [
        {"dow": int(w), "label": _DOW_LABELS[int(w)], "mean": float(g[target].mean())}
        for w, g in d.groupby("dow")
    ]
    monthly = [
        {"month": int(m), "mean": float(g[target].mean())}
        for m, g in d.groupby("month")
    ]
    seasonality = {"hourly": hourly, "weekly": weekly, "monthly": monthly}

    # --- downsampled series for the overview line chart (~600 points) ---
    step = max(1, len(df) // 600)
    sampled = df.iloc[::step]
    series = [
        {"time": t.isoformat(), "value": None if pd.isna(v) else round(float(v), 3)}
        for t, v in zip(sampled["date"], sampled[target])
    ]

    # --- auto-generated observations ---
    observations = _observations(df, numeric, target, meta, summary, correlation, hourly)

    return {
        "dataset": dataset,
        "meta": meta,
        "columns": columns,
        "summary": summary,
        "missing": missing,
        "histogram": histogram,
        "correlation": correlation,
        "seasonality": seasonality,
        "series": series,
        "observations": observations,
    }


def _observations(df, numeric, target, meta, summary, correlation, hourly) -> list[str]:
    obs: list[str] = []
    years = (df["date"].max() - df["date"].min()).days / 365.25
    obs.append(
        f"The series spans {meta['rows']:,} {meta['freq']} records over "
        f"~{years:.1f} years ({meta['start'][:10]} → {meta['end'][:10]})."
    )

    # strongest correlate of the target
    cols = correlation["columns"]
    ti = cols.index(target)
    corr_row = correlation["matrix"][ti]
    ranked = sorted(
        ((cols[i], corr_row[i]) for i in range(len(cols)) if cols[i] != target),
        key=lambda kv: abs(kv[1]), reverse=True,
    )
    if ranked:
        name, val = ranked[0]
        direction = "positively" if val > 0 else "negatively"
        obs.append(
            f"Oil Temperature (OT) is most strongly {direction} correlated with "
            f"{name} (r = {val:.2f})."
        )

    # missingness
    total_missing = sum(s["missing"] for s in summary)
    if total_missing == 0:
        obs.append("The dataset is complete — no missing values in any column.")
    else:
        worst = max(summary, key=lambda s: s["missing_pct"])
        obs.append(
            f"{total_missing:,} missing values overall; most affected column is "
            f"{worst['column']} ({worst['missing_pct']}%)."
        )

    # daily seasonality peak/trough
    if hourly:
        peak = max(hourly, key=lambda h: h["mean"])
        trough = min(hourly, key=lambda h: h["mean"])
        obs.append(
            f"OT shows a daily cycle: highest around {peak['hour']:02d}:00 "
            f"(avg {peak['mean']:.2f}) and lowest around {trough['hour']:02d}:00 "
            f"(avg {trough['mean']:.2f})."
        )

    # spread / outlier hint on the target
    ot_stats = next(s for s in summary if s["column"] == target)
    iqr = ot_stats["p75"] - ot_stats["p25"]
    if iqr > 0 and (ot_stats["max"] - ot_stats["p75"]) > 3 * iqr:
        obs.append(
            "OT has a long upper tail (max far beyond the 75th percentile), "
            "suggesting occasional temperature spikes worth flagging as anomalies."
        )
    return obs


@router.get("")
def list_datasets():
    """List datasets that have a local CSV available."""
    available = [d for d in DATASETS if (_DATA_DIR / f"{d}.csv").exists()]
    return {"datasets": available}


@router.get("/{dataset}")
def get_eda(dataset: str):
    if dataset not in DATASETS:
        raise HTTPException(404, f"Unknown dataset: {dataset}")
    return _compute(dataset)
