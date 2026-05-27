"""
Утилиты для логирования в MLflow.

- Единый эксперимент для всех runs
- Логирование моделей в MLflow Models
- Регистрация датасетов
- Примеры детекции
- Автоматический экспорт в ONNX
"""

import os
import sys
from datetime import datetime
from pathlib import Path

import mlflow


def setup_experiment(experiment_name: str, run_name: str) -> str:
    """
    Создать или переиспользовать эксперимент.
    Все runs будут в одном эксперименте.
    """
    mlflow.set_tracking_uri("http://mlflow:5000")

    experiment = mlflow.set_experiment(experiment_name)
    experiment_id = experiment.experiment_id

    print(f"🏃 Experiment: {experiment_name}")
    print(f"🆔 Experiment ID: {experiment_id}")
    print(f"📝 Run name: {run_name}")

    return experiment_id


def log_model_as_mlflow(model_path: str, model_name: str, run_id: str = None):
    """
    Залогировать модель в MLflow Models registry.
    """
    try:
        import mlflow.pyfunc
        
        class YOLOModelWrapper(mlflow.pyfunc.PythonModel):
            def __init__(self, model_path):
                self.model_path = model_path
                self.model = None
            
            def load_context(self, context):
                from ultralytics import YOLO
                self.model = YOLO(self.model_path)
            
            def predict(self, context, model_input):
                results = self.model(model_input, verbose=False)
                return results[0].boxes.xyxy.tolist()
        
        wrapper = YOLOModelWrapper(model_path)
        
        mlflow.pyfunc.log_model(
            artifact_path=f"model_{model_name}",
            python_model=wrapper,
            registered_model_name=model_name,
        )
        print(f"📦 Model logged: {model_name}")
    except Exception as e:
        print(f"⚠️  Could not log model to MLflow Models: {e}")


def log_onnx_model(onnx_path: str, model_name: str):
    """
    Залогировать ONNX модель в MLflow.
    """
    try:
        import mlflow.onnx
        
        mlflow.onnx.log_model(
            onnx_model=onnx_path,
            artifact_path=f"onnx_{model_name}",
            registered_model_name=f"{model_name}_onnx",
        )
        print(f"📦 ONNX model logged: {model_name}")
    except Exception as e:
        print(f"⚠️  Could not log ONNX model: {e}")


def log_dataset(dataset_path: str, name: str):
    """
    Залогировать датасет в MLflow.
    """
    try:
        import yaml
        with open(dataset_path) as f:
            ds_info = yaml.safe_load(f)

        mlflow.log_param("dataset_name", name)
        mlflow.log_param("dataset_path", dataset_path)
        mlflow.log_param("dataset_classes", ds_info.get("names", {}))
        mlflow.log_param("num_classes", len(ds_info.get("names", {})))

        print(f"📊 Dataset logged: {name}")
    except Exception as e:
        print(f"⚠️  Could not log dataset: {e}")


def log_detection_examples(model, data_yaml: str, num_examples: int = 5):
    """
    Залогировать примеры детекции в MLflow artifacts с bbox и labels.
    """
    try:
        import yaml
        import random
        from pathlib import Path
        import cv2

        with open(data_yaml) as f:
            ds_info = yaml.safe_load(f)

        dataset_root = Path(ds_info["path"])
        val_images_dir = dataset_root / ds_info["val"]

        if not val_images_dir.exists():
            print(f"⚠️  Validation images not found: {val_images_dir}")
            return

        image_files = list(val_images_dir.glob("*.jpg"))
        if not image_files:
            print(f"⚠️  No images found in {val_images_dir}")
            return

        selected = random.sample(image_files, min(num_examples, len(image_files)))

        for idx, img_path in enumerate(selected):
            results = model(str(img_path), verbose=False, conf=0.25)

            # plot() рисует bbox, labels и confidence
            annotated = results[0].plot(
                line_width=2,
                font_size=12,
                labels=True,
                conf=True,
                boxes=True
            )

            output_path = f"/tmp/detection_example_{idx+1}.jpg"
            cv2.imwrite(output_path, annotated)

            mlflow.log_artifact(output_path, artifact_path="detection_examples")

            # Логируем информацию о детекциях
            num_detections = len(results[0].boxes)
            mlflow.log_metric(f"detection_example_{idx+1}_objects", num_detections)
            print(f"📸 Logged detection example {idx+1}/{num_examples} ({num_detections} objects)")

        print(f"✅ Logged {len(selected)} detection examples")
    except Exception as e:
        print(f"⚠️  Could not log detection examples: {e}")


def start_trace(span_name: str):
    """
    Начать трассировку для пайплайна.
    """
    try:
        from mlflow.tracing import trace

        return trace
    except Exception:
        return None


def export_to_onnx(model_path: str, output_name: str) -> str:
    """
    Экспортировать модель в ONNX и залогировать в MLflow.
    """
    try:
        from ultralytics import YOLO

        model = YOLO(model_path)

        onnx_path = f"/weights/{output_name}.onnx"
        model.export(format="onnx", imgsz=320, simplify=True)

        # YOLO создает файл рядом с .pt файлом
        source_onnx = model_path.replace(".pt", ".onnx")
        if os.path.exists(source_onnx):
            import shutil
            shutil.move(source_onnx, onnx_path)
            print(f"📦 ONNX exported: {onnx_path}")

            # Логируем в MLflow
            mlflow.log_artifact(onnx_path, artifact_path="onnx")

            return onnx_path
        else:
            print(f"⚠️  ONNX file not found: {source_onnx}")
            return None
    except Exception as e:
        print(f"⚠️  Could not export to ONNX: {e}")
        return None
