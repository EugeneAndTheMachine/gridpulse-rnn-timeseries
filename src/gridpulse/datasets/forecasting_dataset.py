"""Dataset for forecasting task."""
import numpy as np
import torch
from .base_dataset import BaseTimeSeriesDataset

class ForecastingDataset(BaseTimeSeriesDataset):
    """
    Sliding-window dataset for forecasting.

    Returns (X, y) where:
        X: (input_len, num_features) — past window
        y: (forecast_horizon,) — future target values
    """
    def __init__(
            self, 
            data: np.ndarray,
            input_len: int = 96,
            forecast_horizon: int = 24,
            stride: int = 1,
            target_col_idx: int = -1
    ):
        super().__init__(data, input_len, target_col_idx)
        self.forecast_horizon = forecast_horizon
        self.stride = stride
        self.total_window_len = input_len + forecast_horizon

    def __len__(self):
        return (len(self.data) - self.total_window_len) // self.stride + 1
    
    def __getitem__(self, idx):
        start = idx * self.stride
        end_input = start + self.input_len
        end_target = end_input + self.forecast_horizon

        X = torch.FloatTensor(self.data[start:end_input])
        y = torch.FloatTensor(self.data[end_input:end_target, self.target_col_idx])

        return X, y