"""Compute anomaly scores from forecasting residuals."""
import numpy as np
import torch
from torch.utils.data import DataLoader
from gridpulse.models.base import BaseNNForecaster
from gridpulse.preprocessing.windowing import TimeSeriesDataset

_VALID_AGGREGATIONS = ("mean", "max", "first")


class ResidualScorer:
    """
    Convert a trained forecaster into anomaly scorer.

    Score = |actual - predicted|. Cao → nhiều khả năng anomaly.

    For multi-step forecast, residuals across the horizon are aggregated
    per `aggregation`:
        - "mean":  average residual across the forecast horizon
        - "max":   worst residual across the forecast horizon
        - "first": use only the 1-step-ahead residual (residuals[:, 0])
    """

    def __init__(
        self,
        model: BaseNNForecaster,
        device: torch.device | None = None,
        aggregation: str = "mean",
    ):
        if aggregation not in _VALID_AGGREGATIONS:
            raise ValueError(
                f"Unknown aggregation '{aggregation}'. "
                f"Valid options: {_VALID_AGGREGATIONS}"
            )
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.aggregation = aggregation

    @torch.no_grad()
    def score(self, X: np.ndarray, y: np.ndarray, batch_size: int = 64) -> np.ndarray:
        """
        Args:
            X: (N, input_len, num_features)
            y: (N, forecast_horizon) — actual values

        Returns:
            scores: (N,) anomaly score per window
        """
        dataset = TimeSeriesDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_residuals = []
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            y_pred = self.model(X_batch).cpu().numpy()
            residuals = np.abs(y_batch.numpy() - y_pred)  # (batch, horizon)
            all_residuals.append(residuals)

        residuals = np.concatenate(all_residuals, axis=0)  # (N, horizon)

        if self.aggregation == "mean":
            return residuals.mean(axis=1)
        if self.aggregation == "max":
            return residuals.max(axis=1)
        # "first" — validated in __init__, so this is safe as the fallback
        return residuals[:, 0]
        
    @torch.no_grad()
    def score_per_step(
        self, X: np.ndarray, y: np.ndarray, batch_size: int = 64
    ) -> np.ndarray:
        """Return full residual matrix (N, horizon) — no aggregation.."""
        dataset = TimeSeriesDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_residuals = []
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            y_pred = self.model(X_batch).cpu().numpy()
            residuals = np.abs(y_batch.numpy() - y_pred)  # (batch, horizon)
            all_residuals.append(residuals)

        return np.concatenate(all_residuals, axis=0)  # (N, horizon)