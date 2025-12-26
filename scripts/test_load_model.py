import os
import mlflow

# Use environment variables with localhost as default
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MINIO_ENDPOINT = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "minio")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "minio123")

# Set environment để access MinIO
os.environ.update(
    {
        "MLFLOW_S3_ENDPOINT_URL": MINIO_ENDPOINT,
        "AWS_ACCESS_KEY_ID": AWS_ACCESS_KEY_ID,
        "AWS_SECRET_ACCESS_KEY": AWS_SECRET_ACCESS_KEY,
    }
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
print(f"📍 MLflow Tracking URI: {MLFLOW_TRACKING_URI}")

try:
    model_uri = "models:/road-mark-yolo/Production"
    print(f"Loading: {model_uri}")

    model = mlflow.pyfunc.load_model(model_uri)
    print("✅ Model loaded successfully!")

except Exception as e:
    print(f"❌ Model load failed: {e}")
