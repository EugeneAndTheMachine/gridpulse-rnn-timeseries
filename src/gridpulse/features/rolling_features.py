"""Rolling statistics features."""
import pandas as pd


def add_rolling_features(
    df: pd.DataFrame,
    target_col: str = "OT",
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Add rolling mean and std for target column."""
    if windows is None:
        windows = [6, 12, 24]

    df = df.copy()
    for w in windows:
        df[f"{target_col}_rmean_{w}"] = df[target_col].rolling(w).mean()
        df[f"{target_col}_rstd_{w}"] = df[target_col].rolling(w).std()

    return df