"""Data cleaning functions for missing markers."""

import pandas as pd

def create_missing_mask(df: pd.DataFrame) -> pd.DataFrame:
    """Create a boolean mask indicating missing values in the dataset."""
    mask = df.isnull().astype(int)
    mask.columns = [f"{col}_missing" for col in df.columns]
    return mask
