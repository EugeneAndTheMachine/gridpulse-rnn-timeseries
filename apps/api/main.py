"""FastAPI app entry point."""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from apps.api.config import settings
from apps.api.routes import health, forecast, anomaly, model_metadata, imputation, eda
from gridpulse.serving.model_loader import get_model_cache
from gridpulse.utils.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm-up: pre-load default model on startup."""
    logger.info("Warming up model cache...")
    try:
        cache = get_model_cache()
        cache.get(settings.default_forecast_model, num_features=30, forecast_horizon=24)
        logger.info(f"Cache size: {cache.size}")
    except Exception as e:
        logger.warning(f"Warmup failed: {e}. Models will load on first request.")

    yield  # app runs here

    logger.info("Shutting down...")
    get_model_cache().clear()


app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(health.router, prefix=settings.api_prefix, tags=["health"])
app.include_router(forecast.router, prefix=settings.api_prefix, tags=["forecast"])
app.include_router(anomaly.router, prefix=settings.api_prefix, tags=["anomaly"])
app.include_router(model_metadata.router, prefix=settings.api_prefix, tags=["models"])
app.include_router(imputation.router, prefix=settings.api_prefix, tags=["imputation"])
app.include_router(eda.router, prefix=settings.api_prefix, tags=["eda"])


@app.get("/")
def root():
    return {
        "service": settings.api_title,
        "version": settings.api_version,
        "docs": "/docs",
        "openapi": "/openapi.json",
    }