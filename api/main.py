# app.py
import os
import tempfile
from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import mlflow

# =========================
# CONFIG
# =========================
mlflow.set_tracking_uri("http://mlflow:5000")

os.environ.update({
    "MLFLOW_S3_ENDPOINT_URL": "http://minio:9000",
    "AWS_ACCESS_KEY_ID": "minio",
    "AWS_SECRET_ACCESS_KEY": "minio123",
    "AWS_DEFAULT_REGION": "us-east-1"
})

MODEL_NAME = "road-mark-yolo"
MODEL_STAGE = "Production"

app = FastAPI(title="YOLO Road Mark Detection API")

# =========================
# LOAD MODEL ON STARTUP
# =========================
model = None

@app.on_event("startup")
def load_model():
    global model

    # model_uri = f"models:/{MODEL_NAME}@{MODEL_STAGE}"

    # # tải artifact từ MLflow (thực chất là từ MinIO)
    # model_path = mlflow.artifacts.download_artifacts(model_uri)

    # # nếu MLflow trả về thư mục
    # if os.path.isdir(model_path):
    #     model_path = os.path.join(artifact_dir, "yolo_run", "weights", "best.pt")

    # model = YOLO(model_path)
    model_uri = f"models:/{MODEL_NAME}@production"

    model_dir = mlflow.artifacts.download_artifacts(model_uri)

    model_path = os.path.join(
        model_dir,
        "yolo_run",
        "weights",
        "best.pt"
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"best.pt not found at {model_path}")

    model = YOLO(model_path)

    print(f"✅ Loaded model {MODEL_NAME} ({MODEL_STAGE})")

# =========================
# HEALTH CHECK
# =========================
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

# =========================
# PREDICT ENDPOINT
# =========================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model not loaded"}

    # lưu file tạm
    with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp:
        tmp.write(await file.read())
        image_path = tmp.name

    # inference
    results = model(image_path)

    detections = []
    for box in results[0].boxes:
        detections.append({
            "cls": int(box.cls[0]),
            "confidence": float(box.conf[0]),
            "bbox": box.xyxy[0].tolist()
        })

    return {
        "num_detections": len(detections),
        "detections": detections
    }
