import os
import mlflow

# Set environment để access MinIO
os.environ.update(
    {
        "MLFLOW_S3_ENDPOINT_URL": "http://localhost:9000",
        "AWS_ACCESS_KEY_ID": "minio",
        "AWS_SECRET_ACCESS_KEY": "minio123",
    }
)

mlflow.set_tracking_uri("http://localhost:5000")

try:
    model_uri = "models:/road-mark-yolo/Production"
    print(f"Loading: {model_uri}")

    model = mlflow.pyfunc.load_model(model_uri)
    print("✅ Model loaded successfully!")

except Exception as e:
    print(f"❌ Model load failed: {e}")
