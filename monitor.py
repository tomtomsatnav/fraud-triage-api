import json
import numpy as np
from sklearn.datasets import make_classification

X_train, _ = make_classification(
    n_samples=10000, n_features=8, n_informative=4,
    weights=[0.99, 0.01], random_state=42
)

live_rows = []
with open("logs/predictions.jsonl") as f:
    for line in f:
        record = json.loads(line)
        live_rows.append(record["features"])
X_live = np.array(live_rows)

print("Training data:", X_train.shape)
print("Live data:", X_live.shape)

for i in range(8):
    train_col = X_train[:, i]
    live_col = X_live[:, i]

    train_mean = train_col.mean()
    train_std = train_col.std()
    live_mean = live_col.mean()

    z = abs(live_mean - train_mean) / train_std
    status = "DRIFT" if z > 2 else "ok"

    print(f"feature {i}: train {train_mean: .2f} live {live_mean: .2f} z={z: .2f} {status}")