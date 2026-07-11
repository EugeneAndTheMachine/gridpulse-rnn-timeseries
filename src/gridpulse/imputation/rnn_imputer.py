"""LSTM-based imputer using bidirectional context."""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from gridpulse.imputation.base import BaseImputer
from gridpulse.utils.logger import logger


class BiLSTMImputerNet(nn.Module):
    """
    Bidirectional LSTM: đọc sequence từ cả 2 phía → predict full sequence.
    Input có mask để model biết vị trí nào cần fill.
    """

    def __init__(
        self,
        num_features: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        # Input: features + mask channels
        self.lstm = nn.LSTM(
            input_size=num_features * 2,  # values + mask
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_size * 2, num_features)  # *2 vì bidirectional

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        # x: (batch, seq_len, num_features) — with 0 in NaN positions
        # mask: (batch, seq_len, num_features) — 1 where observed, 0 where missing
        inp = torch.cat([x, mask], dim=-1)
        out, _ = self.lstm(inp)
        return self.fc(out)  # (batch, seq_len, num_features)


class _ImputationDataset(Dataset):
    """Tạo (window, mask, target) tuples cho training."""

    def __init__(
        self,
        data: np.ndarray,
        mask: np.ndarray,
        seq_len: int = 96,
        stride: int = 24,
    ):
        self.data = data
        self.mask = mask
        self.seq_len = seq_len
        self.stride = stride

    def __len__(self):
        return max(0, (len(self.data) - self.seq_len) // self.stride + 1)

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.seq_len
        window = self.data[start:end]
        m = self.mask[start:end]

        # Input: window với NaN → 0, cùng mask
        x_input = np.nan_to_num(window, nan=0.0).astype(np.float32)
        m_input = m.astype(np.float32)
        target = np.nan_to_num(window, nan=0.0).astype(np.float32)

        return (
            torch.from_numpy(x_input),
            torch.from_numpy(m_input),
            torch.from_numpy(target),
        )


class RNNImputer(BaseImputer):
    """
    Train BiLSTM to reconstruct time series with masked-out values.

    Training: mask out random windows during training,
             loss chỉ tính trên positions bị mask.
    Inference: dùng model để fill actual NaN positions.
    """

    def __init__(
        self,
        seq_len: int = 96,
        hidden_size: int = 64,
        num_layers: int = 2,
        max_epochs: int = 30,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        train_mask_ratio: float = 0.15,
        device: str = "auto",
    ):
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.train_mask_ratio = train_mask_ratio

        if device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        self.model: BiLSTMImputerNet | None = None
        self._columns: list[str] = []
        self._means: dict[str, float] = {}
        self._stds: dict[str, float] = {}

    @property
    def name(self) -> str:
        return f"BiLSTM_h{self.hidden_size}"

    def _standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in self._columns:
            df[col] = (df[col] - self._means[col]) / (self._stds[col] + 1e-8)
        return df

    def _inverse_standardize(self, arr: np.ndarray) -> np.ndarray:
        arr = arr.copy()
        for i, col in enumerate(self._columns):
            arr[:, i] = arr[:, i] * self._stds[col] + self._means[col]
        return arr

    def fit(self, df: pd.DataFrame) -> "RNNImputer":
        numeric_df = df.select_dtypes(include="number")
        self._columns = numeric_df.columns.tolist()

        # Standardize (dùng valid values only)
        for col in self._columns:
            self._means[col] = float(numeric_df[col].mean())
            self._stds[col] = float(numeric_df[col].std())

        scaled_df = self._standardize(numeric_df)
        data = scaled_df.values  # (T, F), có thể có NaN
        mask = (~np.isnan(data)).astype(np.float32)

        num_features = data.shape[1]
        self.model = BiLSTMImputerNet(
            num_features=num_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
        ).to(self.device)

        dataset = _ImputationDataset(data, mask, self.seq_len)
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )

        self.model.train()
        for epoch in range(1, self.max_epochs + 1):
            total_loss = 0.0
            n_samples = 0
            for x, m, target in loader:
                x, m, target = (
                    x.to(self.device),
                    m.to(self.device),
                    target.to(self.device),
                )

                # Randomly mask out additional positions for self-supervision
                extra_mask = (
                    torch.rand_like(m) > self.train_mask_ratio
                ).float()
                # Chỉ mask thêm ở positions vốn observed (m=1)
                train_mask = m * extra_mask
                x_masked = x * train_mask

                pred = self.model(x_masked, train_mask)

                # Loss chỉ tính ở positions bị mask (m=1 nhưng train_mask=0)
                loss_mask = m * (1 - extra_mask)
                loss = ((pred - target) ** 2 * loss_mask).sum() / (
                    loss_mask.sum() + 1e-8
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item() * x.size(0)
                n_samples += x.size(0)

            if epoch % 5 == 0 or epoch == 1:
                logger.info(
                    f"[{self.name}] Epoch {epoch}: loss={total_loss / max(n_samples, 1):.6f}"
                )

        return self

    @torch.no_grad()
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Must call fit() before transform()")

        self.model.eval()
        df = df.copy()
        scaled_df = self._standardize(df[self._columns])
        data = scaled_df.values
        mask = (~np.isnan(data)).astype(np.float32)
        data_filled = np.nan_to_num(data, nan=0.0).astype(np.float32)

        # Sliding window inference — average overlapping predictions
        T, F = data.shape
        predictions = np.zeros((T, F), dtype=np.float32)
        counts = np.zeros((T, F), dtype=np.float32)

        stride = self.seq_len // 2
        for start in range(0, T - self.seq_len + 1, stride):
            end = start + self.seq_len
            x = torch.from_numpy(data_filled[start:end]).unsqueeze(0).to(self.device)
            m = torch.from_numpy(mask[start:end]).unsqueeze(0).to(self.device)
            pred = self.model(x, m).squeeze(0).cpu().numpy()

            predictions[start:end] += pred
            counts[start:end] += 1

        # Handle tail
        if T % stride != 0:
            start = T - self.seq_len
            end = T
            x = torch.from_numpy(data_filled[start:end]).unsqueeze(0).to(self.device)
            m = torch.from_numpy(mask[start:end]).unsqueeze(0).to(self.device)
            pred = self.model(x, m).squeeze(0).cpu().numpy()
            predictions[start:end] += pred
            counts[start:end] += 1

        avg_predictions = predictions / np.maximum(counts, 1)

        # Chỉ replace NaN, giữ nguyên observed values
        missing_positions = np.isnan(data)
        filled = np.where(missing_positions, avg_predictions, data)

        # Inverse standardize
        filled = self._inverse_standardize(filled)

        df[self._columns] = filled
        return df