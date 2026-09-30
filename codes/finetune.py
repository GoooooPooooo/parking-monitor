"""
Скрипт fine-tune YOLOv8 на датасете парковок.
Скачивает датасет и дообучает модель.
Метрики и артефакты логируются в MLflow.
"""

import argparse
import os
import sys
import shutil
import time
import psutil
import yaml
import multiprocessing
from pathlib import Path

from ultralytics import YOLO
from ultralytics import settings

# Отключаем встроенную интеграцию MLflow в Ultralytics
settings.update({'mlflow': False})

# Multiprocessing защита для Windows
if os.name == 'nt':
    multiprocessing.freeze_support()
    DEFAULT_WORKERS = 0
else:
    DEFAULT_WORKERS = 8

# MLflow setup
import mlflow

MLFLOW_TRACKING_URI = os.environ.get('MLFLOW_TRACKING_URI', f'file:///{os.path.abspath("mlruns")}')
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


# ==================== ШАГ 1: Скачивание датасета ====================

def download_dataset(dataset_id="anandpanda3/parking-lot-object-detection-in-yolov8"):
    """Скачать датасет парковок с Kaggle."""
    print("=" * 60)
    print("ШАГ 1: Скачивание датасета...")
    print("=" * 60)
    print(f"Dataset: {dataset_id}")

    try:
        import kagglehub
        path = kagglehub.dataset_download(dataset_id)
        print(f"✅ Датасет скачан в: {path}")
        return path
    except ImportError:
        print("Установите kagglehub: pip install kagglehub")
        return None
    except Exception as e:
        print(f"Ошибка скачивания: {e}")
        return None


# ==================== ШАГ 2: Подготовка датасета ====================

def prepare_dataset(dataset_path):
    """Подготовить датасет в формате YOLO."""
    print("\n" + "=" * 60)
    print("ШАГ 2: Подготовка датасета...")
    print("=" * 60)
    print(f"Содержимое: {os.listdir(dataset_path)}")
    return dataset_path


# ==================== ШАГ 3: Создание конфига ====================

def create_config(dataset_path, nc=2, class_names=None):
    """Создать YAML конфиг для YOLO."""
    if class_names is None:
        class_names = ['space-empty', 'space-occupied']
    
    print("\n" + "=" * 60)
    print("ШАГ 3: Создание конфига...")
    print("=" * 60)

    config = {
        'path': os.path.abspath(dataset_path),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': nc,
        'names': class_names
    }

    config_path = 'parking_dataset.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"✅ Конфиг сохранён: {config_path}")
    return config_path


# ==================== ШАГ 4: Утилиты ====================

def log_dataset(data_yaml: str, name: str):
    """Залогировать датасет в MLflow."""
    try:
        with open(data_yaml) as f:
            ds_info = yaml.safe_load(f)

        mlflow.log_param("dataset_name", name)
        mlflow.log_param("dataset_path", ds_info.get("path", ""))
        mlflow.log_param("dataset_classes", ds_info.get("names", {}))
        mlflow.log_param("num_classes", len(ds_info.get("names", {})))
        print(f"📊 Dataset logged: {name}")
    except Exception as e:
        print(f"⚠️  Could not log dataset: {e}")


def log_detection_examples(model, data_yaml: str, num_examples: int = 5):
    """Залогировать примеры детекции в MLflow artifacts."""
    try:
        import cv2
        import random

        with open(data_yaml) as f:
            ds_info = yaml.safe_load(f)

        dataset_root = Path(ds_info["path"])
        val_images_dir = dataset_root / ds_info.get("val", "valid/images")

        if not val_images_dir.exists():
            print(f"⚠️  Validation images not found: {val_images_dir}")
            return

        image_files = list(val_images_dir.glob("*.jpg")) + list(val_images_dir.glob("*.png"))
        if not image_files:
            print(f"⚠️  No images found in {val_images_dir}")
            return

        selected = random.sample(image_files, min(num_examples, len(image_files)))

        for idx, img_path in enumerate(selected):
            results = model(str(img_path), verbose=False, conf=0.25)
            annotated = results[0].plot(line_width=2, font_size=12, labels=True, conf=True, boxes=True)

            output_path = f"detection_example_{idx+1}.jpg"
            cv2.imwrite(output_path, annotated)
            mlflow.log_artifact(output_path, artifact_path="detection_examples")

            num_detections = len(results[0].boxes)
            mlflow.log_metric(f"detection_example_{idx+1}_objects", num_detections)
            print(f"📸 Logged detection example {idx+1} ({num_detections} objects)")

            # Cleanup
            if os.path.exists(output_path):
                os.remove(output_path)

        print(f"✅ Logged {len(selected)} detection examples")
    except Exception as e:
        print(f"⚠️  Could not log detection examples: {e}")


def export_to_onnx(model_path: str, output_name: str) -> str:
    """Экспорт модели в ONNX и логирование в MLflow."""
    try:
        print("\n" + "=" * 60)
        print("ЭКСПОРТ: Конвертация в ONNX...")
        print("=" * 60)

        model = YOLO(model_path)
        onnx_path = model.export(format="onnx", imgsz=320, simplify=True)

        if onnx_path and os.path.exists(onnx_path):
            # Логируем в MLflow
            mlflow.log_artifact(onnx_path, artifact_path="onnx")
            print(f"✅ ONNX exported and logged: {onnx_path}")
            return onnx_path
        else:
            print(f"⚠️  ONNX export failed")
            return None
    except Exception as e:
        print(f"⚠️  Could not export to ONNX: {e}")
        return None


# ==================== ШАГ 5: Fine-tune ====================

def finetune_model(
    config_path,
    model_name='yolov8m.pt',
    epochs=30,
    imgsz=320,
    batch=8,
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    warmup_momentum=0.8,
    patience=10,
    mosaic=0.5,
    mixup=0.1,
    degrees=0.0,
    translate=0.1,
    scale=0.5,
    shear=0.0,
    perspective=0.0,
    flip_lr=0.5,
    flip_ud=0.0,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    optimizer='auto',
    amp=True,
    workers=0,
    name='parking_finetune',
    verbose=True,
    plots=True,
    experiment_name='YOLOv8 Parking Detection',
    export_onnx=True,
):
    """Дообучить YOLOv8 на датасете парковок."""
    print("\n" + "=" * 60)
    print(f"ШАГ 5: Fine-tune {model_name} ({epochs} эпох)...")
    print("=" * 60)

    # Создаём/получаем эксперимент
    experiment = mlflow.set_experiment(experiment_name)
    experiment_id = experiment.experiment_id
    print(f"🏃 Experiment: {experiment_name}")
    print(f"🆔 Experiment ID: {experiment_id}")

    # Загружаем модель
    model = YOLO(model_name)

    # Параметры модели
    params = {
        "model_name": model_name,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "optimizer": optimizer,
        "lr0": lr0,
        "lrf": lrf,
        "momentum": momentum,
        "weight_decay": weight_decay,
        "warmup_epochs": warmup_epochs,
        "warmup_momentum": warmup_momentum,
        "mosaic": mosaic,
        "mixup": mixup,
        "degrees": degrees,
        "translate": translate,
        "scale": scale,
        "shear": shear,
        "perspective": perspective,
        "fliplr": flip_lr,
        "flipud": flip_ud,
        "hsv_h": hsv_h,
        "hsv_s": hsv_s,
        "hsv_v": hsv_v,
        "amp": amp,
        "workers": workers,
        "patience": patience,
    }

    # Callbacks для логирования метрик по эпохам
    def on_train_epoch_end(trainer):
        """Callback для логирования train метрик после каждой эпохи."""
        epoch = trainer.epoch

        if hasattr(trainer, 'tloss') and trainer.tloss is not None:
            tloss = trainer.tloss
            if len(tloss) >= 3:
                mlflow.log_metrics({
                    "train/box_loss": float(tloss[0]),
                    "train/cls_loss": float(tloss[1]),
                    "train/dfl_loss": float(tloss[2]),
                }, step=epoch)
                print(f"  📊 Epoch {epoch}: train losses logged")

    def on_fit_epoch_end(trainer):
        """Callback для логирования val метрик после валидации."""
        metrics = trainer.metrics
        epoch = trainer.epoch

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
            print(f"  📊 Epoch {epoch}: val metrics logged")

    # Добавляем callbacks в модель
    model.add_callback("on_train_epoch_end", on_train_epoch_end)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

    # Запускаем MLflow run
    with mlflow.start_run(experiment_id=experiment_id, run_name=name) as run:
        run_id = run.info.run_id
        print(f"🏃 Run ID: {run_id}")

        # Логируем датасет и параметры (внутри run!)
        log_dataset(config_path, "parking_detection")
        mlflow.log_params(params)

        # System metrics - начальные
        start_time = time.time()
        initial_memory = psutil.virtual_memory()
        initial_disk = psutil.disk_io_counters()

        # Train model
        results = model.train(
            data=config_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            name=name,
            patience=patience,
            lr0=lr0,
            lrf=lrf,
            momentum=momentum,
            weight_decay=weight_decay,
            warmup_epochs=warmup_epochs,
            warmup_momentum=warmup_momentum,
            mosaic=mosaic,
            mixup=mixup,
            degrees=degrees,
            translate=translate,
            scale=scale,
            shear=shear,
            perspective=perspective,
            fliplr=flip_lr,
            flipud=flip_ud,
            hsv_h=hsv_h,
            hsv_s=hsv_s,
            hsv_v=hsv_v,
            optimizer=optimizer,
            amp=amp,
            workers=workers,
            verbose=verbose,
            plots=plots,
        )

        # Training time
        training_time = time.time() - start_time
        mlflow.log_metric("training_time_seconds", training_time)

        # System metrics - финальные
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
        final_metrics = results.results_dict
        mlflow.log_metrics({
            "final/mAP50": float(final_metrics.get("metrics/mAP50(B)", 0)),
            "final/mAP50-95": float(final_metrics.get("metrics/mAP50-95(B)", 0)),
            "final/precision": float(final_metrics.get("metrics/precision(B)", 0)),
            "final/recall": float(final_metrics.get("metrics/recall(B)", 0)),
            "final/box_loss": float(final_metrics.get("train/box_loss", 0)),
            "final/cls_loss": float(final_metrics.get("train/cls_loss", 0)),
        })

        # Log training artifacts
        run_dir = str(model.trainer.save_dir)
        print(f"\n📊 Save directory: {run_dir}")

        artifacts = [
            "weights/best.pt",
            "weights/last.pt",
            "confusion_matrix.png",
            "PR_curve.png",
            "F1_curve.png",
            "results.png",
            "labels.jpg",
            "args.yaml",
        ]

        for artifact in artifacts:
            path = os.path.join(run_dir, artifact)
            if os.path.exists(path):
                mlflow.log_artifact(path, artifact_path="artifacts")
                print(f"  ✅ {artifact}")

        output_path = os.path.join(run_dir, "weights/best.pt")
        print(f"\n✅ Модель сохранена: {output_path}")

        # Log detection examples
        try:
            log_detection_examples(model, config_path, num_examples=5)
        except Exception as e:
            print(f"⚠️ Could not log detection examples: {e}")

        # Export to ONNX
        if export_onnx:
            try:
                export_to_onnx(output_path, name)
            except Exception as e:
                print(f"⚠️ Could not export to ONNX: {e}")

    print(f"\n✅ MLflow run завершён. Run ID: {run_id}")
    return output_path


# ==================== ГЛАВНЫЙ СКРИПТ ====================

def parse_args():
    """Парсинг аргументов командной строки."""
    parser = argparse.ArgumentParser(
        description='Fine-tune YOLOv8 для детекции парковок',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    train = parser.add_argument_group('Параметры обучения')
    train.add_argument('--epochs', type=int, default=30)
    train.add_argument('--batch', type=int, default=8)
    train.add_argument('--imgsz', type=int, default=320)
    train.add_argument('--lr0', type=float, default=0.01)
    train.add_argument('--lrf', type=float, default=0.01)
    train.add_argument('--momentum', type=float, default=0.937)
    train.add_argument('--weight_decay', type=float, default=0.0005)
    train.add_argument('--warmup_epochs', type=float, default=3.0)
    train.add_argument('--warmup_momentum', type=float, default=0.8)
    train.add_argument('--patience', type=int, default=10)

    aug = parser.add_argument_group('Аугментация')
    aug.add_argument('--mosaic', type=float, default=0.5)
    aug.add_argument('--mixup', type=float, default=0.1)
    aug.add_argument('--degrees', type=float, default=0.0)
    aug.add_argument('--translate', type=float, default=0.1)
    aug.add_argument('--scale', type=float, default=0.5)
    aug.add_argument('--shear', type=float, default=0.0)
    aug.add_argument('--perspective', type=float, default=0.0)
    aug.add_argument('--flip_lr', type=float, default=0.5)
    aug.add_argument('--flip_ud', type=float, default=0.0)
    aug.add_argument('--hsv_h', type=float, default=0.015)
    aug.add_argument('--hsv_s', type=float, default=0.7)
    aug.add_argument('--hsv_v', type=float, default=0.4)

    model_grp = parser.add_argument_group('Модель')
    model_grp.add_argument('--model', type=str, default='yolov8m.pt')
    model_grp.add_argument('--optimizer', type=str, default='auto')
    model_grp.add_argument('--amp', action='store_true', default=True)
    model_grp.add_argument('--no_amp', action='store_false', dest='amp')

    data_grp = parser.add_argument_group('Датасет')
    data_grp.add_argument('--dataset', type=str, default=None)
    data_grp.add_argument('--data_path', type=str, default=None)
    data_grp.add_argument('--nc', type=int, default=2)
    data_grp.add_argument('--class_names', type=str, nargs='+', default=['space-empty', 'space-occupied'])

    mlflow_grp = parser.add_argument_group('MLflow')
    mlflow_grp.add_argument('--mlflow', action='store_true', default=True)
    mlflow_grp.add_argument('--no_mlflow', action='store_false', dest='mlflow')
    mlflow_grp.add_argument('--experiment_name', type=str, default='YOLOv8 Parking Detection')
    mlflow_grp.add_argument('--run_name', type=str, default='parking_finetune')

    export_grp = parser.add_argument_group('Экспорт')
    export_grp.add_argument('--export_onnx', action='store_true', default=True)
    export_grp.add_argument('--no_onnx', action='store_false', dest='export_onnx')

    misc = parser.add_argument_group('Прочее')
    misc.add_argument('--workers', type=int, default=DEFAULT_WORKERS)
    misc.add_argument('--verbose', action='store_true', default=True)
    misc.add_argument('--plots', action='store_true', default=True)

    return parser.parse_args()


if __name__ == "__main__":
    if os.name == 'nt':
        multiprocessing.freeze_support()

    args = parse_args()

    print("🚗 Fine-tune YOLOv8 для детекции парковок")
    print("=" * 60)
    print(f"Модель: {args.model}")
    print(f"Epochs: {args.epochs}, Batch: {args.batch}, Imgsz: {args.imgsz}")
    print(f"LR: {args.lr0}, LRF: {args.lrf}, Optimizer: {args.optimizer}")
    print(f"MLflow: {'вкл' if args.mlflow else 'выкл'}")
    print("=" * 60)

    dataset_id = args.dataset or "anandpanda3/parking-lot-object-detection-in-yolov8"

    if args.data_path:
        dataset_path = args.data_path
        print(f"📂 Используем локальный датасет: {dataset_path}")
    else:
        dataset_path = download_dataset(dataset_id)
        if not dataset_path:
            print("\n❌ Не удалось скачать датасет.")
            print("  --data_path 'C:/path/to/dataset'")
            sys.exit(1)

    dataset_path = prepare_dataset(dataset_path)
    config_path = create_config(dataset_path, nc=args.nc, class_names=args.class_names)

    if not args.mlflow:
        print("\n⚠️ MLflow отключён")
        best_model = finetune_model(
            config_path,
            model_name=args.model, epochs=args.epochs, imgsz=args.imgsz,
            batch=args.batch, lr0=args.lr0, lrf=args.lrf,
            momentum=args.momentum, weight_decay=args.weight_decay,
            warmup_epochs=args.warmup_epochs, warmup_momentum=args.warmup_momentum,
            patience=args.patience, mosaic=args.mosaic, mixup=args.mixup,
            degrees=args.degrees, translate=args.translate, scale=args.scale,
            shear=args.shear, perspective=args.perspective,
            flip_lr=args.flip_lr, flip_ud=args.flip_ud,
            hsv_h=args.hsv_h, hsv_s=args.hsv_s, hsv_v=args.hsv_v,
            optimizer=args.optimizer, amp=args.amp, workers=args.workers,
            name=args.run_name, verbose=args.verbose, plots=args.plots,
            experiment_name=args.experiment_name, export_onnx=args.export_onnx,
        )
    else:
        best_model = finetune_model(
            config_path,
            model_name=args.model, epochs=args.epochs, imgsz=args.imgsz,
            batch=args.batch, lr0=args.lr0, lrf=args.lrf,
            momentum=args.momentum, weight_decay=args.weight_decay,
            warmup_epochs=args.warmup_epochs, warmup_momentum=args.warmup_momentum,
            patience=args.patience, mosaic=args.mosaic, mixup=args.mixup,
            degrees=args.degrees, translate=args.translate, scale=args.scale,
            shear=args.shear, perspective=args.perspective,
            flip_lr=args.flip_lr, flip_ud=args.flip_ud,
            hsv_h=args.hsv_h, hsv_s=args.hsv_s, hsv_v=args.hsv_v,
            optimizer=args.optimizer, amp=args.amp, workers=args.workers,
            name=args.run_name, verbose=args.verbose, plots=args.plots,
            experiment_name=args.experiment_name, export_onnx=args.export_onnx,
        )

    print("\n" + "=" * 60)
    print("✅ Fine-tune завершён!")
    print(f"Модель: {best_model}")
    print("=" * 60)
