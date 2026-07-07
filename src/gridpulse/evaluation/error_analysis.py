"""Breakdown errors by time-of-day, day-of-week and season"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def error_by_hour(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: pd.DatetimeIndex,
    model_name: str
):
    """
    Analyze: does the model perform worse at certain hours?

    timestamps should correspond to the START of each prediction window.
    """
    errors = np.abs(y_true - y_pred).mean(axis=1)  # MAE per sample
    hours = timestamps.hour

    hourly_mae = pd.DataFrame({"hour": hours, "mae": errors}).groupby("hour")["mae"].mean()

    fig, ax = plt.subplots(figsize=(10, 4))
    hourly_mae.plot(kind="bar", ax=ax, color="steelblue", alpha=0.7)
    ax.set_title(f"{model_name} — MAE by Hour of Day")
    ax.set_ylabel("MAE")
    ax.axhline(errors.mean(), color="red", linestyle="--", label="Overall MAE")
    ax.legend()
    plt.tight_layout()
    plt.show()

    return hourly_mae