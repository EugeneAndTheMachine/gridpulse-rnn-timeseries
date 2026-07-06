"""Data cleaning functions for time series."""
import pandas as pd
import numpy as np
from gridpulse.utils.logger import logger


def clean_ett(df: pd.DataFrame) -> pd.DataFrame:
    """Clean ETT dataset. ETT is mostly clean, so this is light."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Check and handle duplicates
    dupes = df["date"].duplicated()
    if dupes.any():
        logger.warning(f"Dropping {dupes.sum()} duplicate timestamps")
        df = df.drop_duplicates(subset=["date"], keep="first")

    # Check time continuity — any gaps?
    expected_freq = pd.Timedelta(hours=1)
    time_diff = df["date"].diff()
    gaps = time_diff[time_diff > expected_freq]
    if len(gaps) > 0:
        logger.warning(f"Found {len(gaps)} time gaps")

    return df


def clean_air_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Clean UCI Air Quality dataset. Replace -200 markers with NaN."""
    df = df.copy()

    # Replace -200 with NaN
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace(-200.0, np.nan)

    # Drop rows that are entirely NaN (trailing empty rows)
    df = df.dropna(how="all")

    missing_pct = df[numeric_cols].isnull().mean() * 100
    logger.info(f"Missing % after cleaning:\n{missing_pct.round(1)}")

    return df