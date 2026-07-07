"""GRU forecaster."""
import torch
import torch.nn as nn
from gridpulse.models.base import BaseNNForecaster

class GRUForecaster(BaseNNForecaster):
    """
    GRU encoder → Linear decoder.

    GRU is simpler than LSTM (no cell state, just hidden state)
    with reset gate and update gate. Often similar performance
    to LSTM but faster to train due to fewer parameters.
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

        self.gru = nn.GRU(
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
        return f"GRU_h{self.hidden_size}_L{self.num_layers}"
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, h_n = self.gru(x)

        last_hidden = h_n[-1]
        last_hidden = self.dropout(last_hidden)
        prediction = self.fc(last_hidden)

        return prediction