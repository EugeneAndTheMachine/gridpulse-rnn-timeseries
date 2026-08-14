"""MICE (Multiple Imputation by Chained Equations) via sklearn IterativeImputer."""
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from gridpulse.imputation.base import BaseImputer


class MICEImputer(BaseImputer):
    """
    MICE: iteratively regress each column on all others.
    Slower than KNN nhưng thường accurate hơn cho MAR data.
    """

    def __init__(
        self,
        max_iter: int = 10,
        random_state: int = 42,
        estimator=None,
    ):
        self.max_iter = max_iter
        self.random_state = random_state
        self._imputer = IterativeImputer(
            estimator=estimator or BayesianRidge(),
            max_iter=max_iter,
            random_state=random_state,
        )
        self._columns: list[str] = []

    @property
    def name(self) -> str:
        return f"MICE_iter{self.max_iter}"

    def fit(self, df: pd.DataFrame) -> "MICEImputer":
        numeric_df = df.select_dtypes(include="number")
        self._columns = numeric_df.columns.tolist()
        self._imputer.fit(numeric_df.values)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        imputed = self._imputer.transform(df[self._columns].values)
        df[self._columns] = imputed
        return df