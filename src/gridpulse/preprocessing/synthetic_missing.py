"""Generate synthetic missing data for controlled experiments."""
import numpy as np
import pandas as pd

def inject_mcar(
        df: pd.DataFrame,
        columns: list[str],
        missing_rate: float = 0.1,
        seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Missing Completely At Random.
    Returns (corrupted_df, mask) where mask=True means value was removed.
    """
    rng = np.random.RandomState(seed)
    df_corrupt = df.copy()
    mask = pd.DataFrame(False, index=df.index, columns=columns)

    for col in columns:
        missing_idx = rng.choice(
            df.index,
            size=int(len(df) * missing_rate),
            replace=False
        )
        df_corrupt.loc[missing_idx, col] = np.nan
        mask.loc[missing_idx, col] = True

    return df_corrupt, mask

def inject_block_missing(
        df: pd.DataFrame,
        column: str,
        block_size: int = 24,
        num_blocks: int = 5,
        seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """    
    Remove contiguous blocks — simulates sensor outage.
    block_size=24 with hourly data = 24-hour outage.
    """
    rng = np.random.RandomState(seed)
    df_corrupt = df.copy()
    mask = pd.DataFrame(False, index=df.index, columns=[column])

    max_start = len(df) - block_size
    starts = rng.choice(max_start, size=num_blocks, replace=False)

    for s in starts:
        idx = df.index[s:s + block_size]
        df_corrupt.loc[idx, column] = np.nan
        mask.loc[idx, column] = True

    return df_corrupt, mask