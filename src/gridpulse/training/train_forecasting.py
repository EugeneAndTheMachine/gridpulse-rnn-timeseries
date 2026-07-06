"""Run baseline experiments for ETT dataset with MLFLow for tracking."""
import mlflow
import numpy as np
from gridpulse.evaluation.metrics_forecasting import compute_all_metrics
from gridpulse.utils.logger import logger
from gridpulse.utils.paths import ROOT

mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")

def evaluate_baseline(
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        experiment_name: str = "gridpulse_baselines"
) -> dict:
    """Evaluate a baseline model and log metrics to MLflow."""
    mlflow.set_experiment(experiment_name)

    # Start an MLflow run
    with mlflow.start_run(run_name=model.name):
        # Predict
        y_pred = model.predict(X_test)

        # Compute metrics
        metrics = compute_all_metrics(y_test, y_pred)

        # Log to MLflow
        mlflow.log_param("model_name", model.name)
        mlflow.log_param("input_shape", X_test.shape[1])
        mlflow.log_param("forecast_horizon", y_test.shape[1])
        mlflow.log_param("test_samples", len(X_test))

        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        logger.info(f"Metrics for {model.name}: {metrics}")
        return metrics
