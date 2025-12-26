import os

# Use environment variables with localhost as default
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MINIO_ENDPOINT = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "minio")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "minio123")

# Set environment variables for MLflow S3 backend
os.environ["MLFLOW_S3_ENDPOINT_URL"] = MINIO_ENDPOINT
os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY

import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

print(f"📍 MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
print(f"📍 MinIO Endpoint: {MINIO_ENDPOINT}")

exp = client.get_experiment_by_name("road-mark-yolo")
run = client.search_runs(
    [exp.experiment_id], order_by=["attributes.start_time DESC"], max_results=1
)[0]

model_uri = f"runs:/{run.info.run_id}/model"

res = mlflow.register_model(model_uri=model_uri, name="road-mark-yolo")

# promote
client.transition_model_version_stage(
    name="road-mark-yolo", version=res.version, stage="Production"
)

print("Registered Production:", res.version)
