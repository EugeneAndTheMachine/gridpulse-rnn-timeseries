"""Vanilla RNN forecaster."""
import torch
import torch.nn as nn
from gridpulse.models.base import BaseNNForecaster

class RNNForecaster(BaseNNForecaster):
    """
    Vanilla RNN encoder → Linear decoder.

    Architecture:
        Input (batch, input_len, num_features)
          → RNN layers
          → Take last hidden state (batch, hidden_size)
          → Linear → (batch, forecast_horizon)
    """
    def __init__(
        self,
        num_features: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        forecast_horizon: int = 24,
    ):
        super().__init__(forecast_horizon)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, forecast_horizon)

    @property
    def name(self) -> str:
        return f"VanillaRNN_h{self.hidden_size}_L{self.num_layers}"
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, input_len, num_features)
        output, h_n = self.rnn(x)
        # output: (batch, input_len, hidden_size)
        # h_n: (num_layers, batch, hidden_size)

        # Use the last hidden state from the top layer
        last_hidden = h_n[-1]  # (batch, hidden_size)
        last_hidden = self.dropout(last_hidden)
        prediction = self.fc(last_hidden)  # (batch, forecast_horizon)

        return prediction