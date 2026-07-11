"""Evaluate imputers using mask-and-recover protocol."""
import numpy as np
import pandas as pd
import mlflow
from gridpulse.imputation.base import BaseImputer
from gridpulse.preprocessing.synthetic_missing import (
    inject_mcar,
    inject_block_missing,
)
from gridpulse.utils.logger import logger


def compute_imputation_metrics(
    y_true: np.ndarray,
    y_imputed: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    """
    Args:
        y_true: ground truth values
        y_imputed: values sau khi impute
        mask: True at positions that were hidden

    Returns:
        MAE, RMSE, MAPE trên hidden positions
    """
    mask_bool = mask.astype(bool)
    if mask_bool.sum() == 0:
        return {"mae": 0.0, "rmse": 0.0, "mape": 0.0, "n_masked": 0}

    diff = y_true[mask_bool] - y_imputed[mask_bool]
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))

    # MAPE
    denominator = np.abs(y_true[mask_bool])
    denominator = np.where(denominator < 1e-8, 1e-8, denominator)
    mape = float(np.mean(np.abs(diff) / denominator) * 100)

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "mape": round(mape, 2),
        "n_masked": int(mask_bool.sum()),
    }


def benchmark_imputers(
    df_clean: pd.DataFrame,
    imputers: list[BaseImputer],
    target_columns: list[str],
    missing_configs: list[dict],
    experiment_name: str = "imputation-benchmark",
) -> pd.DataFrame:
    """
    Benchmark based on many imputation scenarios (mask-and-recover).

    missing_configs example:
        [
            {"type": "mcar", "rate": 0.1, "seed": 42},
            {"type": "block", "block_size": 24, "num_blocks": 5, "seed": 42},
        ]
    """
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(experiment_name)

    results = []
    for cfg in missing_configs:
        if cfg["type"] == "mcar":
            df_corrupt, mask_df = inject_mcar(
                df_clean, target_columns, cfg["rate"], seed=cfg.get("seed", 42)
            )
            scenario = f"mcar_{int(cfg['rate']*100)}"
        elif cfg["type"] == "block":
            df_corrupt, mask_df = inject_block_missing(
                df_clean,
                target_columns[0],
                block_size=cfg["block_size"],
                num_blocks=cfg["num_blocks"],
                seed=cfg.get("seed", 42),
            )
            scenario = f"block_{cfg['block_size']}h_x{cfg['num_blocks']}"
        else:
            raise ValueError(f"Unknown missing type: {cfg['type']}")

        # Build full mask array
        mask_full = np.zeros_like(df_clean[target_columns].values, dtype=bool)
        for i, col in enumerate(target_columns):
            if col in mask_df.columns:
                mask_full[:, i] = mask_df[col].values

        y_true = df_clean[target_columns].values

        for imputer in imputers:
            with mlflow.start_run(run_name=f"{imputer.name}_{scenario}"):
                logger.info(f"Running {imputer.name} on {scenario}...")

                df_imputed = imputer.fit_transform(df_corrupt.copy())
                y_imputed = df_imputed[target_columns].values

                metrics = compute_imputation_metrics(y_true, y_imputed, mask_full)

                mlflow.log_param("imputer", imputer.name)
                mlflow.log_param("scenario", scenario)
                for k, v in metrics.items():
                    mlflow.log_metric(k, v)

                results.append({
                    "imputer": imputer.name,
                    "scenario": scenario,
                    **metrics,
                })

                logger.info(f"  {imputer.name} / {scenario}: MAE={metrics['mae']}")

    return pd.DataFrame(results)