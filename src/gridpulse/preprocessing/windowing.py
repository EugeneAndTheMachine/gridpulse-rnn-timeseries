"""Sliding window for smoothing and feature extraction."""
import numpy as np
import torch
from torch.utils.data import Dataset

def create_windows(
        data: np.ndarray,
        input_len: int,
        forecast_horizon: int,
        stride: int = 1,
        target_col_idx: int = -1
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create sliding windows from time series data.

    Args:
        data: shape (T, num_features) — the full time series
        input_len: number of past timesteps as input (e.g. 96)
        forecast_horizon: number of future timesteps to predict (e.g. 24)
        stride: step between consecutive windows (1 = max overlap)
        target_col_idx: which column is the target (-1 = last = OT)

    Returns:
        X: shape (num_windows, input_len, num_features)
        y: shape (num_windows, forecast_horizon)
    """  
    T, F = data.shape
    total_window = input_len + forecast_horizon

    X_list, y_list = [], []
    for start in range(0, T - total_window + 1, stride):
        end_input = start + input_len
        end_target = end_input + forecast_horizon

        X_list.append(data[start:end_input]) # All features for input window
        y_list.append(data[end_input:end_target, target_col_idx]) # Only target column for forecast horizon

    return np.array(X_list), np.array(y_list)


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series data with sliding windows."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)  # shape (num_windows, input_len, num_features)
        self.y = torch.FloatTensor(y)  # shape (num_windows, forecast_horizon

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
