from pydantic import BaseModel, Field
from fastapi import FastAPI
import os
from typing import Annotated
import json
from datetime import datetime, timezone
from pathlib import Path
from monitor import check_drift
from app.scorer import FraudScorer

LOG_PATH = Path("logs/predictions.jsonl")
LOG_PATH.parent.mkdir(exist_ok=True)

THRESHOLD = float(os.getenv("FRAUD_THRESHOLD", "0.2"))
MODEL_PATH = os.getenv("MODEL_PATH", "model_artifact")

scorer = FraudScorer(MODEL_PATH, THRESHOLD)
class ClaimFeatures(BaseModel):
    features: Annotated[list[float], Field(min_length=8, max_length=8)]

app = FastAPI()

@app.get("/health")
def health():
    return {
        "status": "ok",
        "threshold": THRESHOLD
        }
@app.post("/predict")
def predict(payload: ClaimFeatures):
    probability, flagged = scorer.score(payload.features)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "features": payload.features,
        "probability": probability,
        "flagged": flagged,
        "threshold": THRESHOLD,
        "model_alias": "champion",
    }
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(record) + "\n")
    return {
        "fraud_probability": probability,
        "flagged": flagged,
        "threshold": THRESHOLD
    }
    
@app.get("/drift")
def drift():
    results = check_drift()
    return {
        "features": results,
        "any_drift": any(r["drifted"] for r in results),
    }