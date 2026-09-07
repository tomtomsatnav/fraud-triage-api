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

MODEL_PATH = os.getenv("MODEL_PATH", "model_artifact")

# scorer.threshold is the single source of truth: it decides the flag, so it is also
# what /health and /predict report and what the log records. Reading the env into a
# second module-level constant let the two drift apart.
scorer = FraudScorer(MODEL_PATH, float(os.getenv("FRAUD_THRESHOLD", "0.2")))
class ClaimFeatures(BaseModel):
    features: Annotated[list[float], Field(min_length=8, max_length=8)]

app = FastAPI()

@app.get("/health")
def health():
    return {
        "status": "ok",
        "threshold": scorer.threshold
        }
@app.post("/predict")
def predict(payload: ClaimFeatures):
    probability, flagged = scorer.score(payload.features)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "features": payload.features,
        "probability": probability,
        "flagged": flagged,
        "threshold": scorer.threshold,
        "model_alias": "champion",
    }
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(record) + "\n")
    return {
        "fraud_probability": probability,
        "flagged": flagged,
        "threshold": scorer.threshold
    }
    
@app.get("/drift")
def drift():
    results = check_drift()
    return {
        "features": results,
        "any_drift": any(r["drifted"] for r in results),
    }