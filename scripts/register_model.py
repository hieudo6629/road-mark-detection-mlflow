import os

os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"

import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://localhost:5000")
client = MlflowClient()

exp = client.get_experiment_by_name("road-mark-yolo")
run = client.search_runs(
    [exp.experiment_id],
    order_by=["attributes.start_time DESC"],
    max_results=1
)[0]

model_uri = f"runs:/{run.info.run_id}/model"

res = mlflow.register_model(
    model_uri=model_uri,
    name="road-mark-yolo"
)

# promote
client.transition_model_version_stage(
    name="road-mark-yolo",
    version=res.version,
    stage="Production"
)

print("Registered Production:", res.version)
