from fastapi import FastAPI, UploadFile, File, HTTPException, Response
import tempfile
import os
import json
import time
from datetime import datetime
from ultralytics import YOLO
import mlflow
from pathlib import Path
from typing import Dict, List
from mlflow.tracking import MlflowClient

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

app = FastAPI(title="YOLO Road Mark Detection API")

# =========================
# METRICS TRACKING
# =========================
class MetricsTracker:
    def __init__(self):
        self.start_time = time.time()
        self.request_count = 0
        self.success_count = 0
        self.error_count = 0
        self.total_inference_time = 0
        self.detection_counts = {}  # class -> count
        self.last_predictions = []
        
    def record_request(self, success: bool, inference_time: float = 0):
        self.request_count += 1
        if success:
            self.success_count += 1
            self.total_inference_time += inference_time
        else:
            self.error_count += 1
            
    def record_detections(self, predictions: List[Dict]):
        self.last_predictions = predictions
        for pred in predictions:
            class_name = pred.get('name', 'unknown')
            self.detection_counts[class_name] = self.detection_counts.get(class_name, 0) + 1

metrics = MetricsTracker()
model = None

# =========================
# MODEL LOADING
# =========================
@app.on_event("startup")
def load_model():
    global model
    print("🔥 Loading model from MLflow...")
    client = MlflowClient()
    try:
        model_uri = "models:/road-mark-yolo@production"
        print(f"📦 URI: {model_uri}")
        
        model_dir = mlflow.artifacts.download_artifacts(model_uri)
        print(f"📁 Downloaded to: {model_dir}")
        model_version = client.get_model_version_by_alias(
            name="road-mark-yolo",
            alias="production"
        )
        print(f"🏷️ Version {model_version.version} (alias: production)")
        print(f"📊 Run ID: {model_version.run_id}")
        pt_files = list(Path(model_dir).rglob("*.pt"))
        
        if not pt_files:
            model_subdir = Path(model_dir) / "model"
            if model_subdir.exists():
                pt_files = list(model_subdir.rglob("*.pt"))
        
        if not pt_files:
            raise FileNotFoundError("No .pt file found in artifacts")
        
        model_path = str(pt_files[0])
        print(f"✅ Found model: {model_path}")
        
        model = YOLO(model_path)
        print("✅ YOLO model loaded successfully")
        
    except Exception as e:
        print(f"❌ Model load failed: {e}")
        raise

# =========================
# HEALTH & METRICS ENDPOINTS
# =========================
@app.get("/health")
def health():
    return {
        "status": "healthy" if model else "unhealthy",
        "model_loaded": model is not None,
        "uptime_seconds": time.time() - metrics.start_time,
        "requests_total": metrics.request_count
    }

@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus metrics endpoint"""
    uptime = time.time() - metrics.start_time
    
    # Prometheus format
    prometheus_data = f"""# HELP api_uptime_seconds Total uptime of the API
# TYPE api_uptime_seconds gauge
api_uptime_seconds {uptime}

# HELP api_requests_total Total number of requests
# TYPE api_requests_total counter
api_requests_total {metrics.request_count}

# HELP api_successful_requests_total Total successful requests
# TYPE api_successful_requests_total counter
api_successful_requests_total {metrics.success_count}

# HELP api_failed_requests_total Total failed requests
# TYPE api_failed_requests_total counter
api_failed_requests_total {metrics.error_count}

# HELP api_average_inference_time_seconds Average inference time in seconds
# TYPE api_average_inference_time_seconds gauge
"""
    
    # Add average inference time if available
    if metrics.success_count > 0:
        avg_inference = metrics.total_inference_time / metrics.success_count
        prometheus_data += f"api_average_inference_time_seconds {avg_inference}\n"
    else:
        prometheus_data += f"api_average_inference_time_seconds 0\n"
    
    # Add detection metrics by class
    prometheus_data += "\n# HELP api_detections_total Total detections by class\n"
    prometheus_data += "# TYPE api_detections_total counter\n"
    for class_name, count in metrics.detection_counts.items():
        # Sanitize label (Prometheus format)
        safe_class = class_name.replace('"', '').replace('\\', '')
        prometheus_data += f'api_detections_total{{class="{safe_class}"}} {count}\n'
    
    # Model info
    prometheus_data += f"""
# HELP api_model_loaded Whether the model is loaded (1=yes, 0=no)
# TYPE api_model_loaded gauge
api_model_loaded {1 if model else 0}
"""
    
    return Response(content=prometheus_data, media_type="text/plain")

@app.get("/metrics/json")
async def json_metrics():
    """JSON metrics endpoint (for debugging)"""
    return {
        "uptime_seconds": time.time() - metrics.start_time,
        "requests": {
            "total": metrics.request_count,
            "successful": metrics.success_count,
            "failed": metrics.error_count,
            "success_rate": metrics.success_count / metrics.request_count if metrics.request_count > 0 else 0
        },
        "inference": {
            "average_time_seconds": metrics.total_inference_time / metrics.success_count if metrics.success_count > 0 else 0,
            "total_time_seconds": metrics.total_inference_time
        },
        "detections": metrics.detection_counts,
        "model": {
            "loaded": model is not None,
            "last_loaded": datetime.fromtimestamp(metrics.start_time).isoformat()
        }
    }

# =========================
# PREDICTION ENDPOINT
# =========================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not model:
        metrics.record_request(success=False)
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not file.content_type.startswith('image/'):
        metrics.record_request(success=False)
        raise HTTPException(status_code=400, detail="File must be an image")
    
    temp_path = None
    start_time = time.time()
    
    try:
        # Save file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            content = await file.read()
            tmp.write(content)
            temp_path = tmp.name
        
        print(f"🔍 Predicting: {file.filename}")
        
        # Predict with timing
        inference_start = time.time()
        results = model(temp_path)
        inference_time = time.time() - inference_start
        
        # Parse results
        if results and len(results) > 0:
            result_json = results[0].tojson()
            
            if isinstance(result_json, str):
                predictions = json.loads(result_json)
            else:
                predictions = result_json
            
            # Record metrics
            metrics.record_request(success=True, inference_time=inference_time)
            metrics.record_detections(predictions)
            
            return {
                "filename": file.filename,
                "detections": len(predictions),
                "inference_time_seconds": inference_time,
                "predictions": predictions
            }
        else:
            metrics.record_request(success=True, inference_time=inference_time)
            return {
                "filename": file.filename,
                "detections": 0,
                "inference_time_seconds": inference_time,
                "predictions": []
            }
            
    except Exception as e:
        metrics.record_request(success=False)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)