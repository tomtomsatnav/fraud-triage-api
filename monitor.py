import json
import numpy as np
from sklearn.datasets import make_classification
from pathlib import Path

def check_drift(log_path="logs/predictions.jsonl", z_threshold=2.0):
    
    path = Path(log_path)
    if not path.exists() or path.stat().st_size == 0:
        return []

    X_train, _ = make_classification(
        n_samples=10000, n_features=8, n_informative=4,
        weights=[0.99, 0.01], random_state=42
    )

    live_rows = []
    with open(path) as f:
        for line in f:
            record = json.loads(line)
            live_rows.append(record["features"])
    X_live = np.array(live_rows)

    results = []

    for i in range(8):
        train_col = X_train[:, i]
        live_col = X_live[:, i]
        train_mean = train_col.mean()
        train_std = train_col.std()
        live_mean = live_col.mean()
        z = abs(live_mean - train_mean) / train_std
        
        results.append({
            "feature": i,
            "train_mean": float(train_mean),
            "live_mean": float(live_mean),
            "z_score": float(z),
            "drifted": bool(z > z_threshold),
        })

    return results

if __name__ == "__main__":
    for r in check_drift():
        print(r)