from fastapi.testclient import TestClient
from app.main import app, scorer

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
    assert body["threshold"] == scorer.threshold

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

def test_scorer_flags_against_its_own_threshold():
    """FraudScorer decides without the HTTP layer — the point of extracting it."""
    from app.scorer import FraudScorer

    features = [3.2, -1.0, 0.3, 2.4, -0.5, 1.4, 4.0, -0.2]
    probability, _ = FraudScorer("model_artifact", 0.2).score(features)

    # Same model and features, two thresholds either side of the probability:
    # only the threshold moves, so only the flag should.
    below = FraudScorer("model_artifact", probability - 0.01).score(features)
    above = FraudScorer("model_artifact", probability + 0.01).score(features)

    assert below == (probability, True)
    assert above == (probability, False)
