"""Threshold strategies for anomaly detection."""
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod


class BaseThreshold(ABC):
    @abstractmethod
    def fit(self, scores: np.ndarray) -> "BaseThreshold":
        """Fit the threshold strategy to the scores."""
        pass

    @abstractmethod
    def predict(self, scores: np.ndarray) -> np.ndarray:
        """Predict anomalies based on the fitted threshold."""
        pass


class PercentileThreshold(BaseThreshold):
    """Anomaly = score > percentile P của training scores."""

    def __init__(self, percentile: float = 95.0):
        self.percentile = percentile
        self.threshold_: float | None = None

    def fit(self, scores: np.ndarray) -> "PercentileThreshold":
        self.threshold_ = float(np.percentile(scores, self.percentile))
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        if self.threshold_ is None:
            raise RuntimeError("Must fit before predict")
        return scores > self.threshold_

class ZScoreThreshold(BaseThreshold):
    """Anomaly = |score - mean| / std > k."""

    def __init__(self, k: float = 3.0):
        self.k = k
        self.mean_: float | None = None
        self.std_: float | None = None

    def fit(self, scores: np.ndarray) -> "ZScoreThreshold":
        self.mean_ = float(scores.mean())
        self.std_ = float(scores.std())
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        if self.std_ is None or self.std_ == 0:
            return np.zeros_like(scores, dtype=bool)
        z = np.abs(scores - self.mean_) / self.std_
        return z > self.k

class RollingThreshold(BaseThreshold):
    """
    Dynamic threshold: anomaly if score > rolling_mean + k * rolling_std.

    Rolling statistics are computed over PAST points only (via `shift(1)`),
    so a spike does not inflate its own threshold. Suitable for
    non-stationary series.
    """

    def __init__(self, window: int = 168, k: float = 3.0):
        self.window = window
        self.k = k

    def fit(self, scores: np.ndarray) -> "RollingThreshold":
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        s = pd.Series(scores)
        past = s.shift(1)
        rolling_mean = past.rolling(self.window, min_periods=self.window // 4).mean()
        rolling_std = past.rolling(self.window, min_periods=self.window // 4).std()
        threshold = rolling_mean + self.k * rolling_std
        threshold = threshold.bfill()
        return (s > threshold).values
