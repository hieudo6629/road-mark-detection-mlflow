import os
from mlflow.tracking import MlflowClient

# Use environment variable with localhost as default
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
print(f"📍 MLflow Tracking URI: {MLFLOW_TRACKING_URI}")

client.transition_model_version_stage(name="road-mark-yolo", version="1", stage="None")
