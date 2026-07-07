"""Visualization functions for model evaluation."""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_predictions_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    num_samples: int = 3,
    save_path: Path | None = None,
):
    """Plot predicted vs actual for a few test samples."""
    fig, axes = plt.subplots(num_samples, 1, figsize=(14, 3 * num_samples))

    for i in range(num_samples):
        idx = np.random.randint(0, len(y_true))
        axes[i].plot(y_true[idx], label="Actual", color="black", linewidth=2)
        axes[i].plot(y_pred[idx], label="Predicted", color="red", linestyle="--")
        axes[i].set_title(f"Sample {idx}")
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    fig.suptitle(f"{model_name} — Predictions vs Actual", fontsize=14)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_error_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save_path: Path | None = None,
):
    """Histogram of prediction errors."""
    errors = y_pred - y_true  # shape: (N, horizon)
    mean_errors = errors.mean(axis=1)  # average error per sample

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(mean_errors, bins=50, alpha=0.7, color="steelblue")
    axes[0].axvline(0, color="red", linestyle="--")
    axes[0].set_title(f"{model_name} — Error Distribution")
    axes[0].set_xlabel("Mean Prediction Error")

    # Error by forecast step
    step_mae = np.abs(errors).mean(axis=0)
    axes[1].bar(range(len(step_mae)), step_mae, color="steelblue", alpha=0.7)
    axes[1].set_title(f"{model_name} — MAE by Forecast Step")
    axes[1].set_xlabel("Forecast Step (h)")
    axes[1].set_ylabel("MAE")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_training_curves(
    history: dict,
    model_name: str,
    save_path: Path | None = None,
):
    """Plot train/val loss curves."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history["train_loss"], label="Train Loss")
    ax.plot(history["val_loss"], label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(f"{model_name} — Training Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()