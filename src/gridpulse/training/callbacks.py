"""Training callbacks"""
import numpy as np
from pathlib import Path
from gridpulse.utils.logger import logger

class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = np.inf
        self.counter = 0
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    f"Early stopping triggered after {self.counter} epochs "
                    f"without improvement. Best val_loss: {self.best_loss:.6f}"
                )
        return self.should_stop
    
class ModelCheckpoint:
    """Save best model checkpoint based on validation loss."""

    def __init__(self, save_dir: Path, model_name: str):
        self.save_dir = save_dir
        self.model_name = model_name
        self.best_loss = np.inf

    @property
    def path(self) -> Path:
        return self.save_dir / f"{self.model_name}_best.pt"
    
    def __call__(self, model, val_loss: float) -> bool:
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            model.save_checkpoint(self.path) # Model has to be a class of BaseNNForecaster
            logger.info(f"Saved best checkpoint: {self.path} (val_loss={val_loss:.6f})")
            return True
        return False