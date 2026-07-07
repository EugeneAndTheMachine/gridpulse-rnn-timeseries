"""Core training loop for time series forecasting models."""

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import mlflow
import numpy as np

from gridpulse.utils.paths import CHECKPOINTS_DIR
from gridpulse.utils.logger import logger
from gridpulse.training.callbacks import EarlyStopping, ModelCheckpoint

class TimeSeriesTrainer:
    """
    General-purpose trainer for BaseNNForecaster models.

    Usage:
        trainer = TimeSeriesTrainer(model, config)
        history = trainer.fit(train_loader, val_loader)
        predictions = trainer.predict(test_loader)
    """

    def __init__(
        self,
        model: nn.Module,
        config: dict,
        device: torch.device | None = None,
    ):
        self.model = model
        self.config = config

        # Device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device
        
        self.model = self.model.to(self.device)
        logger.info(f"Using device: {self.device}")
        logger.info(f"Model: {self.model.name}, Parameters: {self.model.count_parameters()}")

        # Loss function - MSE for regression tasks
        self.criterion = nn.MSELoss()

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.get("learning_rate", 1e-3)
        )

        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

        # Callbacks
        self.early_stopping = EarlyStopping(
            patience=config.get("patience", 10)
        )
        self.checkpoint = ModelCheckpoint(
            save_dir=CHECKPOINTS_DIR,
            model_name=self.model.name
        )

        # Gradient clipping
        self.gradient_clip = config.get("grad_clip", 1.0)

    def _train_one_epoch(self, train_loader: DataLoader) -> float:
        """Run one training epoch. Returns average loss."""
        self.model.train()
        epoch_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

            self.optimizer.zero_grad()
            y_pred = self.model(X_batch)
            loss = self.criterion(y_pred, y_batch)
            loss.backward()

            # Gradient clipping
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

            self.optimizer.step()
            epoch_loss += loss.item() * X_batch.size(0)

        return epoch_loss / len(train_loader.dataset)
    
    @torch.no_grad()
    def _validate(self, val_loader: DataLoader) -> float:
        """Run validation. Returns average loss."""
        self.model.eval()
        total_loss = 0.0

        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            y_pred = self.model(X_batch)
            loss = self.criterion(y_pred, y_batch)
            total_loss += loss.item() * X_batch.size(0)

        return total_loss / len(val_loader.dataset)
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        experiment_name: str = "gridpulse-forecasting",
    ) -> dict:
        """
        Full training loop with MLflow logging.
        
        Returns training history dict.
        """
        max_epochs = self.config.get("max_epochs", 100)
        history = {"train_loss": [], "val_loss": [], "lr": []}

        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=f"{self.model.name}_train"):
            # Log all config params
            mlflow.log_param("model_name", self.model.name)
            mlflow.log_param("num_parameters", self.model.count_parameters())

            for k, v in self.config.items():
                mlflow.log_param(k, v)

            start_time = time.time()

            for epoch in range(1, max_epochs + 1):
                train_loss = self._train_one_epoch(train_loader)
                val_loss = self._validate(val_loader)
                current_lr = self.optimizer.param_groups[0]["lr"]

                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_loss)
                history["lr"].append(current_lr)

                # Log metrics to MLflow
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)
                mlflow.log_metric("learning_rate", current_lr, step=epoch)

                # LR scheduler step
                self.scheduler.step(val_loss)

                # Logging
                if epoch % 5 == 0 or epoch == 1:
                    logger.info(
                        f"Epoch {epoch}/{max_epochs} | "
                        f"Train Loss: {train_loss:.6f} | "
                        f"Val Loss: {val_loss:.6f} | "
                        f"LR: {current_lr:.2e}"
                    )

                # Checkpointing
                self.checkpoint(self.model, val_loss)
                
                # Early stopping
                if self.early_stopping(val_loss):
                    break

            elapsed = time.time() - start_time
            mlflow.log_metric("training_time_seconds", elapsed)
            mlflow.log_metric("final_epoch", epoch)
            mlflow.log_metric("best_val_loss", self.checkpoint.best_loss)

            logger.info(
                f"Training complete: {epoch} epochs, "
                f"{elapsed:.1f}s, best_val_loss={self.checkpoint.best_loss:.6f}"
            )

        return history
    
    @torch.no_grad()
    def predict(self, test_loader: DataLoader) -> np.ndarray:
        """Generate predictions for test set."""
        # Load best checkpoint
        self.model.load_checkpoint(self.checkpoint.path, self.device)
        self.model.eval()

        predictions = []
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(self.device)
            y_pred = self.model(X_batch)
            predictions.append(y_pred.cpu().numpy())

        return np.concatenate(predictions, axis=0)