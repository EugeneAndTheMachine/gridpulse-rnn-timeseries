"""API configuration via env vars."""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    api_title: str = "GridPulse API"
    api_version: str = "1.0.0"
    api_prefix: str = "/api/v1"

    cors_origins: list[str] = ["http://localhost:8501", "http://localhost:3000"]

    # Model serving
    default_forecast_model: str = "LSTM_h128_L2"
    default_dataset: str = "ETTh1"
    max_forecast_horizon: int = 96
    max_forecast_limit: int = 10000

    class Config:
        env_prefix = "GRIDPULSE_"
        env_file = ".env"


settings = Settings()