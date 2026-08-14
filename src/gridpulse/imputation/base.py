"""Base class for imputers."""
from abc import ABC, abstractmethod
import pandas as pd

class BaseImputer(ABC):
    """
    Base class for all imputation methods.

    Convention:
        - fit(df) learns from training data (some methods are stateless)
        - transform(df) fills NaN values, returns imputed DataFrame
        - fit_transform(df) = fit + transform
    """

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> "BaseImputer":
        pass

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    