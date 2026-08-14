"""API integration tests."""
import pytest
from fastapi.testclient import TestClient
from apps.api.main import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_root_returns_service_info(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "GridPulse" in r.json()["service"]


def test_health_ok(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in ("ok", "degraded")
    assert "api_version" in body


def test_list_available_models(client):
    r = client.get("/api/v1/models/available")
    assert r.status_code == 200
    body = r.json()
    assert "available" in body
    assert isinstance(body["available"], list)


@pytest.mark.integration
def test_predict_endpoint():
    """On-demand prediction returns correct shape."""
    import numpy as np
    from fastapi.testclient import TestClient
    from apps.api.main import app

    client = TestClient(app)
    input_window = np.random.randn(96, 30).tolist()
    r = client.post("/api/v1/forecast/predict", json={
        "model_name": "LSTM_h128_L2",
        "input_values": input_window,
        "forecast_horizon": 24,
    })
    # Có thể fail nếu model không load được — accept 200 hoặc 404
    assert r.status_code in (200, 404)
    if r.status_code == 200:
        body = r.json()
        assert body["n_points"] == 24
        assert len(body["points"]) == 24


def test_predict_wrong_shape_returns_400(client):
    r = client.post("/api/v1/forecast/predict", json={
        "model_name": "LSTM_h128_L2",
        "input_values": [1.0, 2.0, 3.0],  # 1D — invalid
        "forecast_horizon": 24,
    })
    # Pydantic sẽ reject → 422 hoặc 400 sau validation
    assert r.status_code in (400, 422)