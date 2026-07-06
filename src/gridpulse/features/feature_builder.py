"""Orchestrate feature engineering pipeline."""
import pandas as pd
from gridpulse.features.calendar_features import add_calendar_features
from gridpulse.features.rolling_features import add_rolling_features
from gridpulse.features.lag_features import add_lag_features
from gridpulse.utils.logger import logger

def build_features(
        df: pd.DataFrame,
        target_col: str = "OT",
        date_col: str = "date",
        use_lags: bool = True,
        use_rolling: bool = True,
        use_calendar: bool = True
)-> pd.DataFrame:
    """
    Full feature engineering pipeline.
    
    IMPORTANT: Call this BEFORE splitting into train/val/test
    to avoid NaN issues at split boundaries.
    Then split. Then scale (fit on train only).
    """

    df = df.copy()

    if use_calendar:
        df = add_calendar_features(df, date_col=date_col)

    if use_lags:
        df = add_lag_features(df, target_col=target_col)

    if use_rolling:
        df = add_rolling_features(df, target_col=target_col)

    # Drop rows with NaN values that may have been introduced by lag/rolling features
    n_before = len(df)
    df = df.dropna().reset_index(drop=True)
    n_dropped = n_before - len(df)
    logger.info(f"Dropped {n_dropped} rows with NaN from feature creations.")

    return df
