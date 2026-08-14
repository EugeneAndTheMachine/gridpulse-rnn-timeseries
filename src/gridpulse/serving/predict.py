"""Prediction logic — pure functions, no HTTP concerns."""
import numpy as np
import torch


@torch.no_grad()
def predict_from_window(
    model: torch.nn.Module,
    window: np.ndarray,
    device: torch.device | None = None,
) -> np.ndarray:
    """
    Single-sample inference.

    Args:
        model: BaseNNForecaster
        window: (input_len, num_features) numpy array
        device: torch device

    Returns:
        predictions: (forecast_horizon,)
    """
    device = device or next(model.parameters()).device
    x = torch.from_numpy(window.astype(np.float32)).unsqueeze(0).to(device)
    y_pred = model(x).squeeze(0).cpu().numpy()
    return y_pred