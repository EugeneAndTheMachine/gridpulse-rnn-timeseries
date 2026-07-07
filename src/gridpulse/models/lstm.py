"""LSTM forecaster."""
import torch
import torch.nn as nn
from gridpulse.models.base import BaseNNForecaster

class LSTMForecaster(BaseNNForecaster):
    """
    LSTM encoder → Linear decoder.

    Same architecture as RNN but uses LSTM cells.
    LSTM has cell state (long-term memory) + hidden state (short-term),
    making it better at capturing long-range dependencies.
    """

    def __init__(
        self,
        num_features: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        forecast_horizon: int = 24
    ):
        super().__init__(forecast_horizon)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, forecast_horizon)

    @property
    def name(self) -> str:
        return f"LSTM_h{self.hidden_size}_L{self.num_layers}"
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, input_len, num_features)
        output, (h_n, c_n) = self.lstm(x)
        # h_n: (num_layers, batch, hidden_size)
        # c_n: (num_layers, batch, hidden_size) - cell state (long-term memory)

        last_hidden = h_n[-1]  # (batch, hidden_size)
        last_hidden = self.dropout(last_hidden)
        prediction = self.fc(last_hidden)  # (batch, forecast_horizon)

        return prediction
    