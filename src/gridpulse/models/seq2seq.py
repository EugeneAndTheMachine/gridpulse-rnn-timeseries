"""Sequence-to-Sequence forecaster with LSTM."""
import torch
import torch.nn as nn
from gridpulse.models.base import BaseNNForecaster


class Seq2SeqForecaster(BaseNNForecaster):
    """
    Encoder-Decoder LSTM for multi-step forecasting.

    Encoder: processes input sequence → context vector (hidden state)
    Decoder: autoregressively generates forecast, one step at a time
    """

    def __init__(
        self,
        num_features: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        forecast_horizon: int = 24,
        teacher_forcing_ratio: float = 0.0,
    ):
        super().__init__(forecast_horizon)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.teacher_forcing_ratio = teacher_forcing_ratio

        # Encoder
        self.encoder = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        # Decoder — input is 1 value (previous prediction or teacher signal)
        self.decoder = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        self.fc_out = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)

    @property
    def name(self) -> str:
        return f"Seq2Seq_h{self.hidden_size}_L{self.num_layers}"

    def forward(
        self, x: torch.Tensor, y: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, input_len, num_features) — encoder input
            y: (batch, forecast_horizon) — target, used for teacher forcing during training
        """
        # Encode
        _, (h_n, c_n) = self.encoder(x)
        # h_n, c_n: (num_layers, batch, hidden_size)

        # Decoder initial input: last known target value
        decoder_input = x[:, -1, -1].unsqueeze(1).unsqueeze(2)
        # shape: (batch, 1, 1)

        outputs = []
        for t in range(self.forecast_horizon):
            decoder_output, (h_n, c_n) = self.decoder(decoder_input, (h_n, c_n))
            prediction = self.fc_out(self.dropout(decoder_output.squeeze(1)))
            # prediction: (batch, 1)
            outputs.append(prediction)

            # Teacher forcing: use ground truth as next input with probability ratio
            if y is not None and self.training:
                use_teacher = torch.rand(1).item() < self.teacher_forcing_ratio
                if use_teacher:
                    decoder_input = y[:, t].unsqueeze(1).unsqueeze(2)
                else:
                    decoder_input = prediction.unsqueeze(1)
            else:
                decoder_input = prediction.unsqueeze(1)

        return torch.cat(outputs, dim=1)  # (batch, forecast_horizon)