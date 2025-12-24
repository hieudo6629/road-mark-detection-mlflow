import os
import mlflow
from ultralytics import YOLO
from datetime import datetime
import time
from pathlib import Path
import json
import tempfile

# ======================
# MLflow config
# ======================
mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("road-mark-yolo")

os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio:9000"
os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"

MODEL_NAME = "road-mark-yolo"

# ======================
# 1. Train YOLO
# ======================
print("🚀 Starting YOLO training...")
model = YOLO("weights/yolov5n.pt")

results = model.train(
    data="data/data.yaml",
    epochs=1,
    imgsz=50,
    batch=1,
    device="cpu",
    verbose=False
)

print(f"✅ Training completed. Results saved to: {results.save_dir}")

# ======================
# 2. Xác định đường dẫn model
# ======================
save_dir = Path(results.save_dir)
best_pt = save_dir / "weights" / "best.pt"

if not best_pt.exists():
    # Fallback: tìm file .pt trong save_dir
    pt_files = list(save_dir.rglob("*.pt"))
    if pt_files:
        best_pt = pt_files[0]
        print(f"⚠️  Using found model: {best_pt}")
    else:
        raise FileNotFoundError(f"No .pt file found in {save_dir}")

print(f"📦 Model file: {best_pt}")

# ======================
# 3. Tạo folder structure cho MinIO
# ======================
timestamp = int(time.time())
date_folder = datetime.now().strftime("%Y/%m/%d")
minio_folder = f"models/{date_folder}/{timestamp}"

print(f"📁 MinIO folder: {minio_folder}")

# ======================
# 4. Log lên MLflow
# ======================
with mlflow.start_run(run_name=f"yolov5-{timestamp}"):
    # Log params
    mlflow.log_params({
        "epochs": 1,
        "imgsz": 50,
        "batch": 1,
        "device": "cpu",
        "model": "yolov5n",
        "minio_folder": minio_folder,
        "timestamp": timestamp
    })
    
    # Log metrics
    if hasattr(results, "results_dict") and results.results_dict:
        for key, value in results.results_dict.items():
            if isinstance(value, (int, float)):
                clean_key = key.replace("/", "_").replace("(", "").replace(")", "").replace(" ", "_")
                mlflow.log_metric(clean_key, value)
    
    # ======================
    # 5. Log artifact vào MinIO với folder structure
    # ======================
    # Log model file
    mlflow.log_artifact(
        local_path=str(best_pt),
        artifact_path=minio_folder  # Đúng: models/2024/01/15/1234567890/best.pt
    )
    
    # Log metadata
    metadata = {
        "training_date": datetime.now().isoformat(),
        "timestamp": timestamp,
        "model_path": f"{minio_folder}/best.pt",
        "dataset": "data/data.yaml",
        "training_config": {
            "epochs": 1,
            "imgsz": 50,
            "batch": 1,
            "device": "cpu"
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(metadata, f, indent=2)
        temp_meta = f.name
    
    mlflow.log_artifact(temp_meta, artifact_path=minio_folder)
    os.unlink(temp_meta)
    
    # ======================
    # 6. Log pyfunc model (QUAN TRỌNG)
    # ======================
    # Tạo wrapper đơn giản nếu không có file
    class SimpleYOLOWrapper(mlflow.pyfunc.PythonModel):
        def load_context(self, context):
            import torch
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                       path=context.artifacts["model_path"])
        
        def predict(self, context, model_input):
            results = self.model(model_input)
            return results.pandas().xyxy[0].to_dict()
    
    # Đường dẫn artifact trong MLflow
    artifact_uri = f"{minio_folder}/best.pt"
    
    # Log model với registered name
    mlflow.pyfunc.log_model(
        artifact_path="model",  # QUAN TRỌNG: "model" không phải "pyfunc_model"
        python_model=SimpleYOLOWrapper(),
        artifacts={"model_path": artifact_uri},
        pip_requirements=[
            "ultralytics==8.0.196",
            "torch==2.0.1",
            "opencv-python",
            "pillow"
        ],
        registered_model_name=MODEL_NAME  # Tự động register
    )
    
    run_id = mlflow.active_run().info.run_id
    print(f"📝 Run ID: {run_id}")

print("=" * 60)
print("✅ TRAIN + LOG + REGISTER COMPLETED SUCCESSFULLY")
print(f"   Model: {MODEL_NAME}")
print(f"   MinIO: {minio_folder}/best.pt")
print(f"   Run ID: {run_id}")
print("=" * 60)