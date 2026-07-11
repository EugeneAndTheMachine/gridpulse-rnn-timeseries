"""Simple statistical imputers: forward/backward fill, mean, median."""
import pandas as pd
from gridpulse.imputation.base import BaseImputer

class ForwardFillImputer(BaseImputer):
    """Propagate last valid observation forward."""

    @property
    def name(self) -> str:
        return "forward_fill"
    
    def fit(self, df: pd.DataFrame) -> "ForwardFillImputer":
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.ffill().bfill()
    

class MeanImputer(BaseImputer):
    """Fill NaN with mean of the column."""

    def __init__(self):
        self.means: dict[str, float] = {}

    @property
    def name(self) -> str:
        return "Mean"
    
    def fit(self, df: pd.DataFrame) -> "MeanImputer":
        numerical_cols = df.select_dtypes(include="number").columns
        self.means = df[numerical_cols].mean().to_dict()
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col, mean_val in self.means.items():
            if col in df.columns:
                df[col] = df[col].fillna(mean_val)
        return df