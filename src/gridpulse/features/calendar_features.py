"""Time-based features extracted from datetime objects."""

import pandas as pd
import numpy as np

def add_calendar_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Add cyclical calendar features."""

    df = df.copy()
    dt = pd.to_datetime(df[date_col])

    # Raw features
    df["hour"] = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek  # Monday=0,
    df["month"] = dt.dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)  # 1 if weekend, else 0

    # Cyclical encoding - sin/cos transforms
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    return df