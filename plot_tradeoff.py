from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt

X, y = make_classification(
    n_samples=10000, n_features=8, n_informative=4,
    weights=[0.99, 0.01], random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_proba = model.predict_proba(X_test)[:, 1]

thresholds = np.arange(0.05, 0.96, 0.05)
precisions = []
recalls = []
f1s = []

for t in thresholds:
    y_pred = (y_proba >= t).astype(int)
    precisions.append(precision_score(y_test, y_pred, zero_division=0))
    recalls.append(recall_score(y_test, y_pred, zero_division=0))
    f1s.append(f1_score(y_test, y_pred, zero_division=0))
    
best_i = int(np.argmax(f1s))
best_t = thresholds[best_i]

plt.figure(figsize=(9, 5))
plt.plot(thresholds, precisions, label="Precision", marker="o")
plt.plot(thresholds, recalls, label="Recall", marker="o")
plt.plot(thresholds, f1s, label="F1", marker="o")

plt.axvline(best_t, linestyle="--", color="grey")
plt.annotate(
    f"Best F1 = {f1s[best_i]:.2f} at {best_t:.2f}",
    xy=(best_t, f1s[best_i]),
    xytext=(best_t + 0.05, f1s[best_i] + 0.1),
)

plt.xlabel("Decision threshold")
plt.ylabel("Score")
plt.title("Precision / recall trade-off — fraud triage (1.4% positive rate)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("docs/tradeoff.png", dpi=150)
print(f"Best F1 {f1s[best_i]:.3f} at threshold {best_t:.2f}")