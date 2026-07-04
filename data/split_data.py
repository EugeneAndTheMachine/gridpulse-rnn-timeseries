"""Time-aware train/val/test splitting."""
import pandas as pd
from gridpulse.utils.logger import logger

def split_by_time(
        df: pd.DataFrame, 
        date_col: str = "date", 
        train_ratio: float = 0.6,
        val_ratio: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """ 
    Split time series chronologically.

    ETT standard split: 12 months train / 4 months val / 4 months test
    With 17,420 rows (ETTh1): train_ratio=0.6, val_ratio=0.2, test=0.2

    Returns (train, val, test) DataFrames.
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()

    logger.info(
        f"Split: train={len(train)} ({train[date_col].min()} → {train[date_col].max()}), "
        f"val={len(val)}, test={len(test)}"
    )

    return train, val, test