# train.py
from ultralytics import YOLO

model = YOLO("weights/yolov5n.pt")

results = model.train(
    data="data/data.yaml",
    epochs=2,
    imgsz=480,
    batch=2,
    workers=0,
    device="cpu",
    project="runs/detect",
    name="train",
    exist_ok=False
)

print(f"✅ Train done at: {results.save_dir}")
