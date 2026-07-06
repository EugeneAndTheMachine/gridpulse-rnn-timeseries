"""Naive baseline for time series forecasting."""
import numpy as np
from gridpulse.models.base import BaseForecaster


class PersistenceForecaster(BaseForecaster):
    """Naive persistence model that predicts the last observed value."""

    def __init__(self, horizon: int = 24):
        self.horizon = horizon

    def fit(self, X_train, y_train):
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        last_value = X[:, -1, -1]  # (batch,)
        return np.repeat(last_value[:, None], self.horizon, axis=1)

    @property
    def name(self) -> str:
        return "PersistenceForecaster"


class SeasonalNaiveForecaster(BaseForecaster):
    """Repeat the seasonal pattern from the lookback window."""

    def __init__(self, seasonal_period: int = 24, horizon: int = 24):
        self.seasonal_period = seasonal_period
        self.horizon = horizon

    @property
    def name(self) -> str:
        return f"SeasonalNaive_{self.seasonal_period}"

    def fit(self, X_train, y_train):
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        season = X[:, -self.seasonal_period:, -1]  # (batch, seasonal_period)
        reps = (self.horizon // self.seasonal_period) + 1
        tiled = np.tile(season, (1, reps))  # (batch, seasonal_period * reps)
        return tiled[:, :self.horizon]
