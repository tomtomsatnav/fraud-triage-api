from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, f1_score, recall_score
import mlflow
import mlflow.sklearn



THRESHOLD = 0.2

# Named explicitly to match export_model.py. MLflow 3 happens to default to
# sqlite:///mlflow.db relative to the working directory, so this is the store training
# already used -- but relying on a library default to agree with the URI the export
# script hardcodes is a coincidence, not a contract, and it changed once already:
# MLflow 2 defaulted to the ./mlruns file store, which cannot host a registry at all.
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("fraud-triage")

with mlflow.start_run(run_name=f"rf-threshold-{THRESHOLD}"):
    X, y = make_classification(n_samples=10000, n_features=8, n_informative=4, weights=[0.99,0.01], random_state=42)
    print(X.shape, y.sum())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(y_train.sum(), y_test.sum())

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred_low = (y_proba >= THRESHOLD).astype(int)
    mlflow.log_param("threshold", THRESHOLD)
    mlflow.log_metric("precision", precision_score(y_test, y_pred_low))
    mlflow.log_metric("recall", recall_score(y_test, y_pred_low))
    mlflow.log_metric("f1", f1_score(y_test, y_pred_low))
    print(classification_report(y_test, y_pred_low))

    mlflow.sklearn.log_model(
    model,
    name="fraud_model",
    registered_model_name="fraud-triage",
)