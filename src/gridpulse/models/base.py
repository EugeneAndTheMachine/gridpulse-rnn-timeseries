"""Base class for all models."""
from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

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

class BaseNNForecaster(nn.Module, ABC):
    """
    Base for all PyTorch forecasting models.
    
    Convention:
        forward(x) → predictions
        x shape:    (batch, input_len, num_features)
        output:     (batch, forecast_horizon)
    """

    def __init__(self, forecast_horizon: int = 24):
        super().__init__()
        self.forecast_horizon = forecast_horizon

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the model."""
        pass

    def save_checkpoint(self, path: Path) -> None:
        """Save the model's state_dict to a checkpoint file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.state_dict(),
            "model_name": self.name,
            "forecast_horizon": self.forecast_horizon,
        }, path)

    def load_checkpoint(self, path: Path, device: torch.device) -> None:
        """Load the model's state_dict from a checkpoint file."""
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        self.load_state_dict(checkpoint["model_state_dict"])

    def count_parameters(self) -> int:
        """Count the number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)