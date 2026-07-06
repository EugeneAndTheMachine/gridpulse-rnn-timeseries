"""Linear baseline model for time series forecasting."""
import numpy as np
from sklearn.linear_model import Ridge
from gridpulse.models.base import BaseForecaster

class LinearForecaster(BaseForecaster):
    """Flatten input window -> Ridge regression for forecasting -> Multi-step output."""

    def __init__(self, alpha: float = 1.0, horizon: int = 24):
        self.alpha = alpha
        self.horizon = horizon
        self.model = None  # Will be initialized in fit

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        # Flatten: (batch, input_len, features) → (batch, input_len * features)
        X_flat = X_train.reshape(X_train.shape[0], -1)
        self.model = Ridge(alpha=self.alpha)
        self.model.fit(X_flat, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Flatten: (batch, input_len, features) → (batch, input_len * features)
        X_flat = X.reshape(X.shape[0], -1)
        # Predict and reshape to (batch, horizon)
        return self.model.predict(X_flat)
    
    @property
    def name(self) -> str:
        return "Linear_Ridge"