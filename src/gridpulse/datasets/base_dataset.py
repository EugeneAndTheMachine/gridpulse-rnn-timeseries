"""Base dataset with shared logic."""
import numpy as np
from torch.utils.data import Dataset

class BaseTimeSeriesDataset(Dataset):
    """Base class for time series datasets."""

    def __init__(self, data: np.ndarray, input_len: int, target_col_idx: int = -1):
        self.data = data
        self.input_len = input_len
        self.target_col_idx = target_col_idx

    def __len__(self):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def __getitem__(self, idx):
        raise NotImplementedError("Subclasses should implement this method.")
