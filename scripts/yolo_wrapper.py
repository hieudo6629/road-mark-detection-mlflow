import mlflow.pyfunc
from ultralytics import YOLO

class YOLOWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model = YOLO(context.artifacts["model_path"])
    
    def predict(self, context, model_input):
        results = self.model(model_input)
        return [r.tojson() for r in results]