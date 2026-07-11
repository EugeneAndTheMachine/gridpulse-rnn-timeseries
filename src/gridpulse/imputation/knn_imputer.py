"""KNN-based imputer wrapping sklearn."""
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer as SklearnKNNImputer
from gridpulse.imputation.base import BaseImputer


class KNNImputer(BaseImputer):
    """
    KNN imputer: fill NaN bằng weighted average của k neighbors.
    Với time series, nên add lag features trước để KNN "hiểu" temporal context.
    """

    def __init__(self, n_neighbors: int = 5, weights: str = "distance"):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self._imputer = SklearnKNNImputer(
            n_neighbors=n_neighbors, weights=weights
        )
        self._columns: list[str] = []

    @property
    def name(self) -> str:
        return f"KNN_k{self.n_neighbors}"

    def fit(self, df: pd.DataFrame) -> "KNNImputer":
        numeric_df = df.select_dtypes(include="number")
        self._columns = numeric_df.columns.tolist()
        self._imputer.fit(numeric_df.values)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        imputed = self._imputer.transform(df[self._columns].values)
        df[self._columns] = imputed
        return df