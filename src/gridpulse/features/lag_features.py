"""Lag and rolling features for time series."""
import pandas as pd


def add_lag_features(
    df: pd.DataFrame,
    target_col: str = "OT",
    lags: list[int] | None = None,
) -> pd.DataFrame:
    """
    Add lagged values of target as features.
    Default lags: 1h, 2h, 3h, 6h, 12h, 24h (1 day), 168h (1 week)
    """
    if lags is None:
        lags = [1, 2, 3, 6, 12, 24, 168]

    df = df.copy()
    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)

    return df