"""Base class for all models."""
from abc import ABC, abstractmethod
import numpy as np

# Create an abstract base class for models
class BaseForecaster(ABC):
    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit the model to the training data."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the fitted model."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the model."""
        pass