import os
from pathlib import Path
import mlflow
import mlflow.pyfunc
from ultralytics import YOLO

# =========================================================
# CONFIG
# =========================================================
MLFLOW_TRACKING_URI = "http://localhost:5000"
MINIO_ENDPOINT = "http://localhost:9000"

MODEL_NAME = "road-mark-yolo"
EXPERIMENT_NAME = "road-mark-yolo"

os.environ.update({
    "MLFLOW_S3_ENDPOINT_URL": MINIO_ENDPOINT,
    "AWS_ACCESS_KEY_ID": "minio",
    "AWS_SECRET_ACCESS_KEY": "minio123"
})

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# =========================================================
# FIND LATEST YOLO TRAIN RUN
# =========================================================
RUNS_DIR = Path("runs/detect")

train_dirs = [d for d in RUNS_DIR.iterdir() if d.is_dir() and d.name.startswith("train")]
if not train_dirs:
    raise RuntimeError("❌ No YOLO training run found in runs/detect")

latest_train_dir = max(train_dirs, key=lambda d: d.stat().st_mtime)
weights_path = latest_train_dir / "weights" / "best.pt"

if not weights_path.exists():
    raise RuntimeError(f"❌ best.pt not found at {weights_path}")

print(f"✅ Using latest YOLO run: {latest_train_dir}")
print(f"✅ Model weights: {weights_path}")

# =========================================================
# YOLO MLflow WRAPPER
# =========================================================
class YOLOv5Wrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model = YOLO(context.artifacts["model_path"])

    def predict(self, context, inputs):
        """
        inputs: list[str] | np.ndarray | PIL.Image
        """
        results = self.model(inputs)
        return results

# =========================================================
# LOG & REGISTER
# =========================================================
with mlflow.start_run(run_name=latest_train_dir.name):

    # -----------------------------
    # LOG PARAMS (minimal & safe)
    # -----------------------------
    mlflow.log_param("model_type", "yolov5n")
    mlflow.log_param("imgsz", 50)
    mlflow.log_param("batch", 1)
    mlflow.log_param("epochs", "see yolo_run/results.csv")

    # -----------------------------
    # LOG TRAINING ARTIFACTS
    # -----------------------------
    mlflow.log_artifacts(
        local_dir=str(latest_train_dir),
        artifact_path="yolo_run"
    )

    # -----------------------------
    # LOG MODEL (CRITICAL PART)
    # -----------------------------
    mlflow.pyfunc.log_model(
        artifact_path="model",   # ⚠️ MUST BE "model"
        python_model=YOLOv5Wrapper(),
        artifacts={
            "model_path": str(weights_path)
        },
        registered_model_name=MODEL_NAME
    )

print("🎉 Model logged & registered successfully!")
