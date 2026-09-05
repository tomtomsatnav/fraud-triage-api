import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.artifacts.download_artifacts(
    artifact_uri="models:/fraud-triage@champion",
    dst_path="model_artifact",
)