"""Load and cache trained models."""
from __future__ import annotations
import torch

from gridpulse.models.lstm import LSTMForecaster
from gridpulse.models.gru import GRUForecaster
from gridpulse.models.rnn import RNNForecaster
from gridpulse.models.seq2seq import Seq2SeqForecaster
from gridpulse.utils.paths import CHECKPOINTS_DIR
from gridpulse.utils.logger import logger


MODEL_REGISTRY = {
    "VanillaRNN": RNNForecaster,
    "LSTM": LSTMForecaster,
    "GRU": GRUForecaster,
    "Seq2Seq": Seq2SeqForecaster,
}


class ModelCache:
    """In-memory cache — load once, serve many."""

    def __init__(self):
        self._cache: dict[str, torch.nn.Module] = {}
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get(
        self,
        model_name: str,
        num_features: int = 30,
        forecast_horizon: int = 24,
    ) -> torch.nn.Module:
        """Get cached model or load from disk."""
        cache_key = f"{model_name}_h{forecast_horizon}"

        if cache_key in self._cache:
            return self._cache[cache_key]

        # Parse architecture từ model name (VanillaRNN_h128_L2 → VanillaRNN)
        arch = model_name.split("_")[0]
        if arch not in MODEL_REGISTRY:
            raise ValueError(f"Unknown architecture: {arch}. Available: {list(MODEL_REGISTRY)}")

        cls = MODEL_REGISTRY[arch]
        model = cls(
            num_features=num_features,
            hidden_size=128,
            num_layers=2,
            forecast_horizon=forecast_horizon,
        )

        ckpt_path = CHECKPOINTS_DIR / f"{model_name}_best.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        model.load_checkpoint(ckpt_path, self._device)
        model.to(self._device).eval()

        self._cache[cache_key] = model
        logger.info(f"Loaded model {model_name} into cache (device={self._device})")
        return model

    def list_available(self) -> list[str]:
        """List models có checkpoint on disk."""
        if not CHECKPOINTS_DIR.exists():
            return []
        return [
            p.stem.replace("_best", "")
            for p in CHECKPOINTS_DIR.glob("*_best.pt")
        ]

    def clear(self):
        """Free memory."""
        self._cache.clear()

    @property
    def size(self) -> int:
        return len(self._cache)


# Singleton
_model_cache: ModelCache | None = None


def get_model_cache() -> ModelCache:
    global _model_cache
    if _model_cache is None:
        _model_cache = ModelCache()
    return _model_cache