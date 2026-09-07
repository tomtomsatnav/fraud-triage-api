from fastapi.testclient import TestClient
from app.main import app, THRESHOLD

client = TestClient(app)

def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_predict_returns_probability_and_flag():
    response = client.post("/predict", json={"features": [0.1] * 8})
    assert response.status_code == 200
    body = response.json()
    assert "fraud_probability" in body
    assert 0 <= body["fraud_probability"] <= 1

def test_predict_returns_wrong_feature_count():
    response = client.post("/predict", json={"features": [0.1] * 7})
    assert response.status_code == 422

def test_predict_includes_threshold_used():
    response = client.post("/predict", json={"features": [0.1] * 8})
    body = response.json()
    assert body["threshold"] == THRESHOLD

def test_predict_flag_agrees_with_threshold():
    response = client.post("/predict", json={"features": [0.1] * 8})
    body = response.json()
    assert body["flagged"] == (body["fraud_probability"] >= body["threshold"])

def test_drift_returns_summary():
    response = client.get("/drift")
    assert response.status_code == 200
    body = response.json()
    assert "any_drift" in body
    assert isinstance(body["features"], list)