"""
Обучение моделей детектора YOLOv8 на датасете COCO airplane.

Поддерживаемые варианты:
- baseline: базовая модель YOLOv8n (23 слоя, 3 детекторных головы)
- mod1: модификация 1 (17 слоёв, 2 детекторных головы)
- mod2: модификация 2 (12 слоёв, 1 детекторная голова)

Логирование в MLflow:
- Уникальный experiment ID (timestamp)
- MLflow Models для регистрации модели
- Dataset с классами
- Метрики: mAP50, mAP50-95, precision, recall
- Артефакты: best.pt, confusion matrix, PR curve, results.png
"""

import argparse
import os
import sys

sys.path.insert(0, "/ml")
from mlflow_utils import setup_experiment, log_model_as_mlflow, log_dataset, log_detection_examples, export_to_onnx
from prepare_coco import ensure_coco_airplane

import mlflow
from ultralytics import YOLO


def ensure_data_available(data: str) -> str:
    """Проверить наличие датасета, при необходимости скачать и подготовить."""
    # data может быть путём к dataset.yaml или к директории
    if data.endswith("dataset.yaml"):
        data_dir = os.path.dirname(data)
    else:
        data_dir = data

    yaml_path = os.path.join(data_dir, "dataset.yaml")
    if os.path.exists(yaml_path):
        train_img = os.path.join(data_dir, "images", "train2017")
        val_img = os.path.join(data_dir, "images", "val2017")
        if os.path.isdir(train_img) and os.path.isdir(val_img):
            if len(os.listdir(train_img)) > 0 and len(os.listdir(val_img)) > 0:
                return yaml_path

    print(f"⚠ Датасет не обнаружен в {data_dir}")
    print("Автоматическое скачивание и подготовка...")
    ensure_coco_airplane(output_dir=data_dir)
    return yaml_path


MODEL_CONFIGS = {
    "baseline": {
        "source": "yolov8n.pt",
        "model_type": "yolov8n",
        "architecture": "23_layers_3_detection_heads",
        "default_name": "yolov8n_baseline",
    },
    "mod1": {
        "source": "/ml/models/yolov8n_mod1.yaml",
        "model_type": "yolov8n_mod1",
        "architecture": "17_layers_2_detection_heads",
        "default_name": "yolov8n_mod1",
        "modification": "trained_variant_1",
    },
    "mod2": {
        "source": "/ml/models/yolov8n_mod2.yaml",
        "model_type": "yolov8n_mod2",
        "architecture": "12_layers_1_detection_head",
        "default_name": "yolov8n_mod2",
        "modification": "trained_variant_2",
    },
}


def train(
    variant: str = "baseline",
    data: str = "/ml/data/coco_airplane/dataset.yaml",
    epochs: int = 100,
    imgsz: int = 320,
    batch: int = 16,
    name: str = None,
    project: str = "mlruns",
    device: str = "0",
    pretrained: bool = True,
):
    """Обучение модели детектора."""
    config = MODEL_CONFIGS[variant]

    if name is None:
        name = config["default_name"]

    # Убедиться что датасет доступен
    data = ensure_data_available(data)

    # Отключить встроенную интеграцию MLflow в Ultralytics
    from ultralytics import settings
    settings.update({'mlflow': False})

    experiment_id = setup_experiment("yolov8_airplane_detection", name)

    # Для baseline можно отключить предобученные веса
    if variant == "baseline" and not pretrained:
        model = YOLO(config["source"])
        # Переопределить на инициализацию без предобученных весов
        model = YOLO("yolov8n.yaml")
    else:
        model = YOLO(config["source"])

    with mlflow.start_run(experiment_id=experiment_id, run_name=name) as run:
        run_id = run.info.run_id

        # Log dataset info
        log_dataset(data, "coco_airplane")

        # Log model parameters
        params = {
            "model_type": config["model_type"],
            "architecture": config["architecture"],
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": batch,
            "device": device,
            "workers": 0,
        }
        if "modification" in config:
            params["modification"] = config["modification"]
        if variant == "baseline":
            params["pretrained"] = pretrained

        mlflow.log_params(params)

        # Train model with tracing
        import time
        import psutil

        start_time = time.time()

        # Получить начальные system metrics
        initial_memory = psutil.virtual_memory()
        initial_disk = psutil.disk_io_counters()

        with mlflow.start_span(name="model_training") as span:
            span.set_attribute("model_type", config["model_type"])
            span.set_attribute("epochs", epochs)
            span.set_attribute("batch_size", batch)

            # Добавить callbacks для логирования метрик по эпохам
            def on_train_epoch_end(trainer):
                """Callback для логирования train метрик после каждой эпохи обучения."""
                epoch = trainer.epoch

                # Логировать train losses (используем tloss - средние за эпоху)
                if hasattr(trainer, 'tloss') and trainer.tloss is not None:
                    tloss = trainer.tloss
                    if len(tloss) >= 3:
                        mlflow.log_metrics({
                            "train/box_loss": float(tloss[0]),
                            "train/cls_loss": float(tloss[1]),
                            "train/dfl_loss": float(tloss[2]),
                        }, step=epoch)

            def on_fit_epoch_end(trainer):
                """Callback для логирования val метрик после валидации."""
                metrics = trainer.metrics
                epoch = trainer.epoch

                # Логировать метрики валидации
                val_metrics = {}
                if "val/box_loss" in metrics:
                    val_metrics["val/box_loss"] = float(metrics["val/box_loss"])
                if "val/cls_loss" in metrics:
                    val_metrics["val/cls_loss"] = float(metrics["val/cls_loss"])
                if "val/dfl_loss" in metrics:
                    val_metrics["val/dfl_loss"] = float(metrics["val/dfl_loss"])
                if "metrics/mAP50(B)" in metrics:
                    val_metrics["val/mAP50"] = float(metrics["metrics/mAP50(B)"])
                if "metrics/mAP50-95(B)" in metrics:
                    val_metrics["val/mAP50-95"] = float(metrics["metrics/mAP50-95(B)"])
                if "metrics/precision(B)" in metrics:
                    val_metrics["val/precision"] = float(metrics["metrics/precision(B)"])
                if "metrics/recall(B)" in metrics:
                    val_metrics["val/recall"] = float(metrics["metrics/recall(B)"])

                if val_metrics:
                    mlflow.log_metrics(val_metrics, step=epoch)

            # Добавить callbacks в модель
            model.add_callback("on_train_epoch_end", on_train_epoch_end)
            model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

            results = model.train(
                data=data,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                name=name,
                project=project,
                save=True,
                plots=True,
                exist_ok=True,
                device=device,
                workers=0,
            )

            training_time = time.time() - start_time
            span.set_attribute("training_time_seconds", training_time)
            mlflow.log_metric("training_time_seconds", training_time)

        # Логировать финальные system metrics
        final_memory = psutil.virtual_memory()
        final_disk = psutil.disk_io_counters()

        mlflow.log_metric("system/peak_memory_mb", final_memory.used / 1024 / 1024)
        mlflow.log_metric("system/memory_percent", final_memory.percent)
        if initial_disk and final_disk:
            disk_read_mb = (final_disk.read_bytes - initial_disk.read_bytes) / 1024 / 1024
            disk_write_mb = (final_disk.write_bytes - initial_disk.write_bytes) / 1024 / 1024
            mlflow.log_metric("system/disk_read_mb", disk_read_mb)
            mlflow.log_metric("system/disk_write_mb", disk_write_mb)

        # Log final metrics
        metrics = results.results_dict
        mlflow.log_metrics({
            "final/mAP50": float(metrics.get("metrics/mAP50(B)", 0)),
            "final/mAP50-95": float(metrics.get("metrics/mAP50-95(B)", 0)),
            "final/precision": float(metrics.get("metrics/precision(B)", 0)),
            "final/recall": float(metrics.get("metrics/recall(B)", 0)),
            "final/box_loss": float(metrics.get("train/box_loss", 0)),
            "final/cls_loss": float(metrics.get("train/cls_loss", 0)),
            "final/dfl_loss": float(metrics.get("train/dfl_loss", 0)),
        })

        # Log training artifacts
        run_dir = os.path.join(project, name)
        artifacts = [
            "weights/best.pt",
            "weights/last.pt",
            "confusion_matrix.png",
            "confusion_matrix_normalized.png",
            "PR_curve.png",
            "P_curve.png",
            "R_curve.png",
            "F1_curve.png",
            "results.png",
            "labels.jpg",
            "labels_correlogram.jpg",
        ]

        for artifact in artifacts:
            path = os.path.join(run_dir, artifact)
            if os.path.exists(path):
                mlflow.log_artifact(path)

    # После завершения with block - логируем в явный run_id
    # Copy weights and log as MLflow Model
    weights_path = os.path.join(project, name, "weights/best.pt")
    if os.path.exists(weights_path):
        os.makedirs("/weights", exist_ok=True)
        import shutil
        dst = f"/weights/{name}.pt"
        shutil.copy2(weights_path, dst)
        print(f"Model saved to {dst}")

        # Log as MLflow Model
        try:
            with mlflow.start_run(run_id=run_id):
                log_model_as_mlflow(dst, name)
        except Exception as e:
            print(f"Could not log model: {e}")

        # Log detection examples
        try:
            with mlflow.start_run(run_id=run_id):
                log_detection_examples(model, data, num_examples=5)
        except Exception as e:
            print(f"Could not log detection examples: {e}")

        # Export to ONNX
        try:
            with mlflow.start_run(run_id=run_id):
                export_to_onnx(dst, name)
        except Exception as e:
            print(f"Could not export to ONNX: {e}")

    print(f"Training complete. Experiment ID: {experiment_id}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Обучение модели детектора YOLOv8",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python train_detector.py --variant baseline
  python train_detector.py --variant mod1 --epochs 50
  python train_detector.py --variant mod2 --device cpu
  python train_detector.py --variant baseline --no-pretrained
        """,
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="baseline",
        choices=["baseline", "mod1", "mod2"],
        help="Вариант модели: baseline (YOLOv8n), mod1 (17 слоев), mod2 (12 слоев)",
    )
    parser.add_argument("--data", type=str, default="/ml/data/coco_airplane/dataset.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--name", type=str, default=None, help="Имя эксперимента (по умолчанию зависит от варианта)")
    parser.add_argument("--device", type=str, default="0", help="0 для GPU, 'cpu' для CPU")
    parser.add_argument("--no-pretrained", action="store_true", help="Обучить с нуля без предобученных весов (только для baseline)")
    args = parser.parse_args()

    train(
        variant=args.variant,
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        device=args.device,
        pretrained=not args.no_pretrained,
    )
