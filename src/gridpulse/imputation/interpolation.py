"""Interpolation-based imputers."""
import pandas as pd
from gridpulse.imputation.base import BaseImputer


class LinearInterpolationImputer(BaseImputer):
    """Linear interpolation giữa các giá trị valid."""

    @property
    def name(self) -> str:
        return "LinearInterp"

    def fit(self, df: pd.DataFrame) -> "LinearInterpolationImputer":
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include="number").columns
        df = df.copy()
        df[numeric_cols] = df[numeric_cols].interpolate(
            method="linear", limit_direction="both"
        )
        return df


class SplineInterpolationImputer(BaseImputer):
    """Cubic spline interpolation — smoother than linear."""

    def __init__(self, order: int = 3):
        self.order = order

    @property
    def name(self) -> str:
        return f"Spline_order{self.order}"

    def fit(self, df: pd.DataFrame) -> "SplineInterpolationImputer":
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include="number").columns
        df = df.copy()
        # Spline cần đủ nhiều điểm valid, có thể fail nên fallback về linear
        try:
            df[numeric_cols] = df[numeric_cols].interpolate(
                method="spline", order=self.order, limit_direction="both"
            )
        except (ValueError, Exception):
            df[numeric_cols] = df[numeric_cols].interpolate(
                method="linear", limit_direction="both"
            )
        return df