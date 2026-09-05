from pydantic import BaseModel
from fastapi import FastAPI
import os
import mlflow.sklearn
from typing import Annotated
from pydantic import Field
import json
from datetime import datetime, timezone
from pathlib import Path

LOG_PATH = Path("logs/predictions.jsonl")
LOG_PATH.parent.mkdir(exist_ok=True)

THRESHOLD = float(os.getenv("FRAUD_THRESHOLD", "0.2"))
MODEL_PATH = os.getenv("MODEL_PATH", "model_artifact")
model = mlflow.sklearn.load_model(MODEL_PATH)

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
    probability = float(model.predict_proba([payload.features])[0][1])
    flagged = probability >= THRESHOLD
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
        "flagged": flagged
    }