import os
import mlflow
from ultralytics import YOLO

# Config
mlflow.set_tracking_uri("http://mlflow:5000")
os.environ.update({
    "MLFLOW_S3_ENDPOINT_URL": "http://minio:9000",
    "AWS_ACCESS_KEY_ID": "minio",
    "AWS_SECRET_ACCESS_KEY": "minio123"
})

MODEL_NAME = "road-mark-yolo"
MODEL_STAGE = "Production"

def load_model():
    try:
        # Load từ Model Registry
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        model = mlflow.pyfunc.load_model(model_uri)
        print("✅ Model loaded from MLflow")
        return model
    except:
        # Fallback: load trực tiếp từ YOLO
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        
        versions = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
        run_id = versions[0].run_id
        
        # Download artifact
        local_path = mlflow.artifacts.download_artifacts(
            artifact_uri=f"runs:/{run_id}/model"
        )
        
        # Load bằng YOLO
        yolo_model = YOLO(f"{local_path}/best.pt")
        print("✅ Model loaded directly")
        return yolo_model

if __name__ == "__main__":
    model = load_model()
    print("Ready for prediction" if model else "Failed")