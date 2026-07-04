import pandas as pd
from gridpulse.utils.logger import logger

ETT_EXPECTED_COLUMNS = ["date", "HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
ETT_NUMERIC_COLUMNS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]

def validate_ett(df: pd.DataFrame) -> bool:
    """Validate ETT dataset schema and basic quality."""
    errors = []

    # Check columns
    missing_cols = set(ETT_EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing columns: {missing_cols}")

    # Check no nulls (ETT should be complete)
    null_counts = df[ETT_NUMERIC_COLUMNS].isnull().sum()
    if null_counts.any():
        errors.append("Dates are not monotonically increasing or there are null values in numeric columns.")

    # Check no duplicated timestamps
    if df["date"].duplicated().any():
        errors.append(f"Found {df['date'].duplicated().sum()} duplicate dates")

    # Checking reasonable value range for OT
    if df["OT"].min() < -50 or df["OT"].max() > 100:
        errors.append(f"OT values are out of expected range: min={df['OT'].min()}, max={df['OT'].max()}")

    if errors:
        for e in errors:
            logger.error(e)
        return False

    logger.info(f"ETT validation passed: {len(df)} rows, {df['date'].min()} to {df['date'].max()}")
    return True