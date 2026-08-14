"""
Backfill DB với forecasts từ trained models.

Usage:
    python scripts/backfill_forecasts.py --model LSTM_h128_L2 --dataset ETTh1
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from gridpulse.data.split_time_series import split_by_time
from gridpulse.database.connection import session_scope
from gridpulse.database.repositories import (
    ForecastRepository,
    ModelRegistryRepository,
)
from gridpulse.features.feature_builder import build_features
from gridpulse.models.lstm import LSTMForecaster
from gridpulse.models.gru import GRUForecaster
from gridpulse.models.rnn import RNNForecaster
from gridpulse.preprocessing.cleaning import clean_ett
from gridpulse.preprocessing.scaling import ScalerWrapper
from gridpulse.preprocessing.windowing import create_windows, TimeSeriesDataset
from gridpulse.utils.paths import RAW_DIR, CHECKPOINTS_DIR
from gridpulse.utils.logger import logger


MODEL_CLASSES = {
    "LSTM_h128_L2": LSTMForecaster,
    "GRU_h128_L2": GRUForecaster,
    "VanillaRNN_h128_L2": RNNForecaster,
}


def load_model(model_name: str, num_features: int, horizon: int):
    """Load trained checkpoint."""
    cls = MODEL_CLASSES[model_name]
    model = cls(
        num_features=num_features,
        hidden_size=128,
        num_layers=2,
        forecast_horizon=horizon,
    )
    ckpt_path = CHECKPOINTS_DIR / f"{model_name}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    model.load_checkpoint(ckpt_path, torch.device("cpu"))
    model.eval()
    logger.info(f"Loaded {model_name} from {ckpt_path}")
    return model


def prepare_test_data(dataset_path: Path, input_len: int, horizon: int):
    """Reproduce Phase 2 pipeline: raw → features → scale → windows."""
    df = pd.read_csv(dataset_path)
    df = clean_ett(df)
    df = build_features(df, target_col="OT")

    train_df, val_df, test_df = split_by_time(df)
    feature_cols = [c for c in train_df.columns if c != "date"]

    scaler = ScalerWrapper(StandardScaler(), feature_cols)
    _ = scaler.fit_transform(train_df)
    test_scaled = scaler.transform(test_df)

    X_test, y_test = create_windows(
        test_scaled[feature_cols].values,
        input_len=input_len,
        forecast_horizon=horizon,
        target_col_idx=-1,
    )
    # Timestamps for each window's forecast START
    # window_i predicts test_df["date"][input_len + i : input_len + i + horizon]
    test_dates = pd.to_datetime(test_df["date"].values)
    forecast_start_times = test_dates[input_len:input_len + len(X_test)]

    return X_test, y_test, forecast_start_times, feature_cols


@torch.no_grad()
def predict_all(model, X_test: np.ndarray, batch_size: int = 64) -> np.ndarray:
    dataset = TimeSeriesDataset(X_test, np.zeros((len(X_test), model.forecast_horizon)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds = []
    for X_batch, _ in loader:
        preds.append(model(X_batch).cpu().numpy())
    return np.concatenate(preds, axis=0)  # (N, horizon)


def build_forecast_records(
    predictions: np.ndarray,
    actuals: np.ndarray,
    start_times: pd.DatetimeIndex,
    model_name: str,
    dataset: str,
    target_col: str = "OT",
) -> list[dict]:
    """Convert (N, horizon) arrays into flat rows for DB insert."""
    records = []
    N, H = predictions.shape

    for i in range(N):
        start_time = start_times[i]
        for step in range(H):
            forecast_time = start_time + pd.Timedelta(hours=step)
            records.append({
                "time": forecast_time.to_pydatetime().replace(tzinfo=timezone.utc),
                "model_name": model_name,
                "dataset": dataset,
                "target_col": target_col,
                "horizon_step": step + 1,
                "predicted": float(predictions[i, step]),
                "actual": float(actuals[i, step]),
            })
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_CLASSES.keys()))
    parser.add_argument("--dataset", default="ETTh1")
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--input-len", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=1000, help="DB insert batch")
    args = parser.parse_args()

    # 1. Prepare test data
    dataset_path = RAW_DIR / "ett" / f"{args.dataset}.csv"
    X_test, y_test, start_times, feature_cols = prepare_test_data(
        dataset_path, args.input_len, args.horizon
    )
    logger.info(f"Test set: {len(X_test)} windows")

    # 2. Load model
    model = load_model(args.model, num_features=len(feature_cols), horizon=args.horizon)

    # 3. Predict
    logger.info("Running predictions...")
    predictions = predict_all(model, X_test)

    # 4. Build records
    records = build_forecast_records(
        predictions, y_test, start_times,
        model_name=args.model, dataset=args.dataset,
    )
    logger.info(f"Built {len(records)} forecast records")

    # 5. Bulk insert với batching
    with session_scope() as session:
        repo = ForecastRepository(session)
        total = 0
        for i in range(0, len(records), args.batch_size):
            batch = records[i:i + args.batch_size]
            total += repo.bulk_insert(batch, on_conflict="update")
        logger.info(f"Inserted/updated {total} forecast rows")

    # 6. Register model
    with session_scope() as session:
        reg_repo = ModelRegistryRepository(session)
        existing = reg_repo.get(args.model, "1.0")
        if existing is None:
            reg_repo.register(
                model_name=args.model,
                version="1.0",
                task="forecasting",
                architecture=args.model.split("_")[0],
                input_len=args.input_len,
                forecast_horizon=args.horizon,
                num_features=len(feature_cols),
                num_parameters=model.count_parameters(),
                checkpoint_path=f"models/checkpoints/{args.model}_best.pt",
                metrics={
                    "mae": float(np.mean(np.abs(predictions - y_test))),
                    "rmse": float(np.sqrt(np.mean((predictions - y_test) ** 2))),
                },
            )
            logger.info(f"Registered model {args.model} v1.0")


if __name__ == "__main__":
    main()