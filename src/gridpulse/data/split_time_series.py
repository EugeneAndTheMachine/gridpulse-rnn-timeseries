"""Time-aware train/validation/test splitting."""

import pandas as pd

from gridpulse.utils.logger import logger


def split_by_time(
    df: pd.DataFrame,
    date_col: str = "date",
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a time series DataFrame chronologically."""
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Expected train_ratio > 0, val_ratio > 0, and train_ratio + val_ratio < 1")

    df = df.sort_values(date_col).reset_index(drop=True) if date_col in df.columns else df.reset_index(drop=True)

    n_rows = len(df)
    train_end = int(n_rows * train_ratio)
    val_end = int(n_rows * (train_ratio + val_ratio))

    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()

    logger.info(f"Split by time: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test
