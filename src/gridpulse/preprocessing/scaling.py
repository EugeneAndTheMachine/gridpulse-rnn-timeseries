"""Scalers for time series features."""
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from dataclasses import dataclass
import pickle
from pathlib import Path

@dataclass
class ScalerWrapper:
    """Wrapper to save/load scaler with column info."""
    scaler: StandardScaler | MinMaxScaler
    columns: list[str]

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[self.columns] = self.scaler.fit_transform(df[self.columns])
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data using a previously fitted scaler."""
        df = df.copy()
        df[self.columns] = self.scaler.transform(df[self.columns])
        return df
    
    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform - critical for interpreting model outputs."""
        df = df.copy()
        df[self.columns] = self.scaler.inverse_transform(df[self.columns])
        return df
    
    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump({"scaler": self.scaler, "columns": self.columns}, f)

    @classmethod
    def load(cls, path: Path) -> "ScalerWrapper":
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(scaler=data["scaler"], columns=data["columns"])
