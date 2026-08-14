"""Database tests — chạy với DB thật, mark là integration."""
import pytest
from datetime import datetime, timezone, timedelta

from gridpulse.database.connection import session_scope
from gridpulse.database.repositories import (
    ForecastRepository, AnomalyRepository, ModelRegistryRepository,
)


@pytest.mark.integration
def test_forecast_bulk_insert_and_query():
    """Insert forecasts, query back, check counts."""
    now = datetime.now(timezone.utc)
    records = [
        {
            "time": now + timedelta(hours=i),
            "model_name": "TEST_MODEL",
            "dataset": "TEST_DATASET",
            "target_col": "OT",
            "horizon_step": (i % 24) + 1,
            "predicted": 20.0 + i * 0.1,
            "actual": 20.0 + i * 0.1 + 0.05,
        }
        for i in range(100)
    ]

    with session_scope() as session:
        repo = ForecastRepository(session)
        n_inserted = repo.bulk_insert(records)
        assert n_inserted == 100

        # Query back
        results = repo.get_forecasts(model_name="TEST_MODEL", limit=200)
        assert len(results) == 100

        # Cleanup
        from gridpulse.database.models import Forecast
        session.query(Forecast).filter(
            Forecast.model_name == "TEST_MODEL"
        ).delete()


@pytest.mark.integration
def test_upsert_updates_existing():
    """Insert same key twice → update, not duplicate."""
    now = datetime.now(timezone.utc)
    record = {
        "time": now, "model_name": "TEST_UPSERT",
        "dataset": "TEST", "target_col": "OT",
        "horizon_step": 1, "predicted": 10.0, "actual": None,
    }

    with session_scope() as session:
        repo = ForecastRepository(session)
        repo.bulk_insert([record])
        repo.bulk_insert([{**record, "predicted": 20.0, "actual": 15.0}])

        results = repo.get_forecasts(model_name="TEST_UPSERT")
        assert len(results) == 1
        assert results[0].predicted == 20.0
        assert results[0].actual == 15.0

        from gridpulse.database.models import Forecast
        session.query(Forecast).filter(
            Forecast.model_name == "TEST_UPSERT"
        ).delete()


@pytest.mark.integration
def test_model_registry_uniqueness():
    """Duplicate (model_name, version) should fail."""
    with session_scope() as session:
        repo = ModelRegistryRepository(session)
        repo.register(
            model_name="TEST_MODEL_REG",
            version="1.0",
            task="forecasting",
            architecture="LSTM",
            checkpoint_path="/tmp/fake.pt",
        )

    with pytest.raises(Exception):
        with session_scope() as session:
            repo = ModelRegistryRepository(session)
            repo.register(
                model_name="TEST_MODEL_REG",
                version="1.0",
                task="forecasting",
                architecture="LSTM",
                checkpoint_path="/tmp/fake.pt",
            )

    # Cleanup
    with session_scope() as session:
        from gridpulse.database.models import ModelRegistry
        session.query(ModelRegistry).filter(
            ModelRegistry.model_name == "TEST_MODEL_REG"
        ).delete()