"""
Обучение модели классификатора типов самолётов (FGVC Aircraft).

Использует ResNet18 из torchvision, предобученную на ImageNet,
и дообучает на датасете FGVC Aircraft (100 классов).

Логирование в MLflow:
- Уникальный experiment ID
- Метрики: accuracy, loss, top1, top5
- System metrics: CPU, memory, disk I/O
- Traces: training span
- Артефакты: best_model.pth, confusion matrix, results charts
- Экспорт в ONNX
"""

import argparse
import os
import sys
import time
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

sys.path.insert(0, "/ml")
from mlflow_utils import setup_experiment, log_dataset
from download_fgvc_aircraft import ensure_dataset

import mlflow


def log_classifier_model_as_mlflow(model_path: str, model_name: str):
    """
    Залогировать модель классификатора в MLflow Models registry.
    """
    try:
        import mlflow.pyfunc

        class ClassifierModelWrapper(mlflow.pyfunc.PythonModel):
            def __init__(self, model_path):
                self.model_path = model_path
                self.model = None
                self.class_names = None
                self.img_size = None

            def load_context(self, context):
                checkpoint = torch.load(self.model_path, map_location="cpu")
                self.class_names = checkpoint.get("class_names", [])
                self.img_size = checkpoint.get("img_size", 224)
                num_classes = checkpoint.get("num_classes", len(self.class_names))
                
                self.model = create_model(num_classes)
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.model.eval()

            def predict(self, context, model_input):
                """
                model_input: PIL Image или numpy array
                Возвращает: dict с predicted class, confidence и top-5 predictions
                """
                from torchvision import transforms
                
                transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(self.img_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])
                
                # Если input PIL Image
                if hasattr(model_input, 'convert'):
                    tensor = transform(model_input).unsqueeze(0)
                else:
                    # numpy array
                    from PIL import Image
                    img = Image.fromarray(model_input)
                    tensor = transform(img).unsqueeze(0)
                
                with torch.no_grad():
                    output = self.model(tensor)
                    probabilities = torch.nn.functional.softmax(output, dim=1)
                    top5_probs, top5_indices = probabilities.topk(5, dim=1)
                
                results = {
                    "predicted_class": self.class_names[top5_indices[0][0]],
                    "confidence": top5_probs[0][0].item(),
                    "top5": [
                        {
                            "class": self.class_names[idx],
                            "confidence": prob.item()
                        }
                        for idx, prob in zip(top5_indices[0], top5_probs[0])
                    ]
                }
                
                return results

        wrapper = ClassifierModelWrapper(model_path)

        mlflow.pyfunc.log_model(
            artifact_path=f"model_{model_name}",
            python_model=wrapper,
            registered_model_name=model_name,
        )
        print(f"📦 Classifier model logged: {model_name}")
    except Exception as e:
        print(f"⚠️  Could not log classifier model to MLflow Models: {e}")
        import traceback
        traceback.print_exc()


def ensure_data_available(data_dir: str) -> str:
    """Проверить наличие датасета, при необходимости скачать и подготовить."""
    print(f"  🔍 Проверка директории: {data_dir}")
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    if os.path.isdir(train_dir) and os.path.isdir(val_dir):
        # Проверить что есть поддиректории с классами
        train_classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
        if train_classes:
            print(f"  ✓ Датасет найден: {len(train_classes)} классов")
            return data_dir

    print(f"  ⚠ Датасет не обнаружен в {data_dir}")
    print("  📥 Автоматическое скачивание и подготовка...")
    return ensure_dataset(output_dir=data_dir)


def get_dataloaders(data_dir: str, batch_size: int = 32, img_size: int = 224):
    """Создать DataLoaders для train/val."""
    print("  🔧 Настройка transforms...")
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    print("  ✓ Transforms настроены")

    print("  📂 Загрузка train dataset...")
    train_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "train"), train_transform
    )
    print(f"  ✓ Train dataset: {len(train_dataset)} изображений")

    print("  📂 Загрузка val dataset...")
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), val_transform)
    print(f"  ✓ Val dataset: {len(val_dataset)} изображений")

    print("  🔄 Создание DataLoaders...")
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    print("  ✓ DataLoaders созданы")

    return train_loader, val_loader, train_dataset.classes, val_dataset.classes


def create_model(num_classes: int):
    """Создать ResNet18 для классификации."""
    print("  📥 Загрузка предобученных весов ResNet18 (ImageNet)...")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    print("  ✓ Веса загружены")
    print("  🔧 Замена FC слоя...")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    print(f"  ✓ FC слой заменён на {num_classes} классов")
    return model


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Progress bar (simple, no tqdm)
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(loader)}] Loss: {loss.item():.4f}")

    return running_loss / total, 100.0 * correct / total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    top5_correct = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Top-5 accuracy
            _, top5_pred = outputs.topk(5, 1, True, True)
            top5_correct += top5_pred.eq(labels.view(-1, 1)).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return (
        running_loss / total,
        100.0 * correct / total,
        100.0 * top5_correct / total,
        all_preds,
        all_labels,
    )


def log_confusion_matrix(all_preds, all_labels, class_names, save_path="/tmp/confusion_matrix.png"):
    """Создать и сохранить confusion matrix."""
    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

        cm = confusion_matrix(all_labels, all_preds)
        fig, ax = plt.subplots(figsize=(20, 20))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(ax=ax, cmap="Blues", xticks_rotation="vertical")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        return True
    except Exception as e:
        print(f"Could not create confusion matrix: {e}")
        return False


def log_classification_examples(model, data_dir, class_names, device, num_examples: int = 5):
    """
    Залогировать примеры классификации в MLflow artifacts с predicted labels и confidence.
    Сначала детектирует самолёт через YOLO, затем классифицирует его тип.
    """
    try:
        import random
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image
        from torchvision import transforms
        from ultralytics import YOLO
        import cv2

        # Загрузить YOLO детектор
        detector_path = "/weights/baseline.pt"
        if not os.path.exists(detector_path):
            print(f"⚠️  Detector not found at {detector_path}, skipping detection")
            return
        
        print(f"🔍 Загрузка YOLO детектора: {detector_path}")
        detector = YOLO(detector_path)

        # Собрать все пути к изображениям из val директории
        val_dir = os.path.join(data_dir, "val")
        if not os.path.exists(val_dir):
            print(f"⚠️  Val directory not found: {val_dir}")
            return

        image_paths = []
        image_labels = []
        
        for class_idx, class_name in enumerate(class_names):
            class_dir = os.path.join(val_dir, class_name)
            if os.path.exists(class_dir):
                for img_file in os.listdir(class_dir):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        image_paths.append(os.path.join(class_dir, img_file))
                        image_labels.append(class_idx)

        if len(image_paths) == 0:
            print("⚠️  No validation images found")
            return

        # Выбрать случайные изображения
        num_to_select = min(num_examples, len(image_paths))
        indices = random.sample(range(len(image_paths)), num_to_select)

        model.eval()
        
        # Трансформ для предсказания (нормализованный)
        eval_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        for idx, i in enumerate(indices):
            img_path = image_paths[i]
            true_label = image_labels[i]
            
            # Загрузить оригинальное изображение
            img_original = Image.open(img_path).convert('RGB')
            img_cv2 = cv2.imread(img_path)
            img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
            
            # Детекция самолёта через YOLO
            det_results = detector(img_path, verbose=False, conf=0.25)
            has_detection = len(det_results[0].boxes) > 0
            
            # Crop detected airplane для классификации
            if has_detection:
                box = det_results[0].boxes[0]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                
                # Crop airplane
                airplane_crop = img_original.crop((x1, y1, x2, y2))
                
                # Подготовить тензор для классификации
                airplane_tensor = eval_transform(airplane_crop)
            else:
                # Если самолёт не обнаружен, используем всё изображение
                airplane_crop = img_original
                airplane_tensor = eval_transform(img_original)
                confidence = 0.0
            
            # Сделать предсказание типа самолёта
            with torch.no_grad():
                input_tensor = airplane_tensor.unsqueeze(0).to(device)
                output = model(input_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                top5_probs, top5_indices = probabilities.topk(5, dim=1)
            
            # Получить предсказанный класс
            pred_label = top5_indices[0][0].item()
            pred_confidence = top5_probs[0][0].item()
            
            # Определить имена классов до рисования
            true_class_name = class_names[true_label] if true_label < len(class_names) else f"Class {true_label}"
            pred_class_name = class_names[pred_label] if pred_label < len(class_names) else f"Class {pred_label}"
            
            # Нарисовать изображение с bbox и подписями
            fig, ax = plt.subplots(figsize=(12, 10))
            ax.imshow(img_cv2)
            ax.axis('off')
            
            # Нарисовать bbox с типом самолёта
            if has_detection:
                from matplotlib.patches import Rectangle
                rect = Rectangle((x1, y1), x2-x1, y2-y1,
                               linewidth=3, edgecolor='lime',
                               facecolor='none', linestyle='-')
                ax.add_patch(rect)

                # Подпись к bbox: предсказанный и реальный тип
                is_correct = (pred_label == true_label)
                label_color = 'lime' if is_correct else 'orange'
                
                # Двухстрочная подпись: предсказание + реальное значение
                pred_line = f"Pred: {pred_class_name} {pred_confidence:.2f}"
                true_line = f"True: {true_class_name}"
                
                # Рисуем предсказание
                ax.text(x1, y1-10, pred_line,
                       fontsize=11, fontweight='bold',
                       color=label_color,
                       bbox=dict(boxstyle='round,pad=0.3',
                                facecolor='black', alpha=0.8))
                
                # Рисуем реальное значение ниже
                ax.text(x1, y1+25, true_line,
                       fontsize=10, fontweight='bold',
                       color='white',
                       bbox=dict(boxstyle='round,pad=0.3',
                                facecolor='navy', alpha=0.7))
            
            # Заголовок с результатом классификации
            status_text = "✓ CORRECT" if pred_label == true_label else "✗ WRONG"
            ax.set_title(status_text, fontsize=14, fontweight='bold', pad=20,
                        color='lime' if pred_label == true_label else 'red')
            
            # Добавить Top-5 predictions внизу слева
            top5_text = "Top-5 Predictions:\n"
            for j, (prob, idx_cls) in enumerate(zip(top5_probs[0], top5_indices[0])):
                class_name = class_names[idx_cls.item()] if idx_cls.item() < len(class_names) else f"Class {idx_cls.item()}"
                top5_text += f"{j+1}. {class_name}: {prob.item():.2%}\n"
            
            ax.text(0.02, 0.02, top5_text.strip(), transform=ax.transAxes,
                    fontsize=11, verticalalignment='bottom',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.85),
                    fontfamily='monospace', color='white')
            
            plt.tight_layout()
            output_path = f"/tmp/classification_example_{idx+1}.jpg"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            # Логировать в MLflow
            mlflow.log_artifact(output_path, artifact_path="classification_examples")
            
            # Логировать метрики
            is_correct = (pred_label == true_label)
            mlflow.log_metric(f"example_{idx+1}_correct", int(is_correct))
            mlflow.log_metric(f"example_{idx+1}_confidence", pred_confidence)
            mlflow.log_metric(f"example_{idx+1}_detected", int(has_detection))
            if has_detection:
                mlflow.log_metric(f"example_{idx+1}_det_confidence", confidence)
            
            status = "✓" if is_correct else "✗"
            det_status = "🎯" if has_detection else "❌"
            print(f"📸 Logged example {idx+1}/{num_examples} {status} {pred_confidence:.2%} {det_status}")

        print(f"✅ Logged {num_to_select} classification examples with detection")
    except Exception as e:
        print(f"⚠️  Could not log classification examples: {e}")
        import traceback
        traceback.print_exc()


def export_to_onnx(model_path: str, output_name: str, num_classes: int, img_size: int = 224) -> str:
    """Экспортировать модель классификатора в ONNX."""
    try:
        # Загрузить checkpoint
        checkpoint = torch.load(model_path, map_location="cpu")

        # Создать модель
        model = create_model(num_classes)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        # Экспорт в ONNX
        onnx_path = f"/weights/{output_name}.onnx"
        dummy_input = torch.randn(1, 3, img_size, img_size)

        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )

        print(f"ONNX exported: {onnx_path}")
        return onnx_path
    except Exception as e:
        print(f"Could not export to ONNX: {e}")
        import traceback
        traceback.print_exc()
        return None


def train(
    data_dir: str = "/ml/data/fgvc_aircraft_imagefolder",
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 0.001,
    img_size: int = 224,
    name: str = "classifier_resnet18",
    device_str: str = "cuda",
):
    """Обучение классификатора с логированием в MLflow."""
    print("=" * 70)
    print("НАЧАЛО ОБУЧЕНИЯ КЛАССИФИКАТОРА")
    print("=" * 70)

    # Normalize device string: "gpu" -> "cuda"
    if device_str.lower() == "gpu":
        device_str = "cuda"
    
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"✓ Device: {device}")

    # Убедиться что датасет доступен
    print("\n📂 Проверка датасета...")
    data_dir = ensure_data_available(data_dir)
    print(f"✓ Датасет готов: {data_dir}")

    print("\n🔧 Настройка MLflow эксперимента...")
    experiment_id = setup_experiment("yolov8_airplane_detection", name)
    print(f"✓ Experiment ID: {experiment_id}")

    print("\n📊 Загрузка данных...")
    train_loader, val_loader, train_classes, val_classes = get_dataloaders(
        data_dir, batch_size, img_size
    )
    num_classes = len(train_classes)
    print(f"✓ Классов: {num_classes}")
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")

    print("\n🏗️  Создание модели ResNet18...")
    model = create_model(num_classes).to(device)
    print(f"✓ Модель создана и перемещена на {device}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )
    print("✓ Optimizer и scheduler настроены")

    best_val_acc = 0.0

    # Создать директорию для артефактов (как в train_detector.py)
    run_dir = os.path.join("mlruns", name)
    os.makedirs(run_dir, exist_ok=True)

    print("\n🚀 Запуск MLflow run...")
    with mlflow.start_run(experiment_id=experiment_id, run_name=name) as run:
        run_id = run.info.run_id
        print(f"✓ Run ID: {run_id}")

        # Log dataset info
        print("\n📝 Логирование параметров в MLflow...")
        mlflow.log_params({
            "model_type": "resnet18",
            "architecture": "ResNet18 + FC head",
            "pretrained": "ImageNet",
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "img_size": img_size,
            "num_classes": num_classes,
            "device": str(device),
            "dataset": "FGVC Aircraft",
            "dataset_classes": num_classes,
        })
        print("✓ Параметры залогированы")

        # Training loop with tracing
        import psutil

        start_time = time.time()
        initial_memory = psutil.virtual_memory()
        initial_disk = psutil.disk_io_counters()

        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []
        val_top5s = []

        print("\n" + "=" * 70)
        print("НАЧАЛО ОБУЧЕНИЯ")
        print("=" * 70)

        with mlflow.start_span(name="classifier_training") as span:
            span.set_attribute("model_type", "resnet18")
            span.set_attribute("epochs", epochs)
            span.set_attribute("batch_size", batch_size)
            span.set_attribute("num_classes", num_classes)

            for epoch in range(epochs):
                epoch_start = time.time()

                train_loss, train_acc = train_epoch(
                    model, train_loader, criterion, optimizer, device
                )
                val_loss, val_acc, val_top5, _, _ = validate(
                    model, val_loader, criterion, device
                )
                scheduler.step(val_loss)

                epoch_time = time.time() - epoch_start

                # Log per-epoch metrics
                mlflow.log_metrics({
                    "train/loss": train_loss,
                    "train/accuracy": train_acc,
                    "val/loss": val_loss,
                    "val/accuracy": val_acc,
                    "val/top5_accuracy": val_top5,
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/epoch_time": epoch_time,
                }, step=epoch)

                train_losses.append(train_loss)
                train_accs.append(train_acc)
                val_losses.append(val_loss)
                val_accs.append(val_acc)
                val_top5s.append(val_top5)

                print(
                    f"Epoch {epoch + 1}/{epochs}: "
                    f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% "
                    f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}% "
                    f"top5={val_top5:.2f}% lr={optimizer.param_groups[0]['lr']:.6f} "
                    f"time={epoch_time:.1f}s"
                )

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    # Сохраняем в директорию как в train_detector.py
                    weights_dir = os.path.join(run_dir, "weights")
                    os.makedirs(weights_dir, exist_ok=True)
                    save_path = os.path.join(weights_dir, "best.pth")
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "class_names": train_classes,
                            "num_classes": num_classes,
                            "img_size": img_size,
                            "epoch": epoch,
                            "val_acc": val_acc,
                        },
                        save_path,
                    )
                    print(f"  Best model saved: {save_path}")
                    
                    # Также сохраняем в /weights/ для совместимости
                    os.makedirs("/weights", exist_ok=True)
                    dst_path = "/weights/classifier.pth"
                    shutil.copy2(save_path, dst_path)

            training_time = time.time() - start_time
            span.set_attribute("training_time_seconds", training_time)
            mlflow.log_metric("training_time_seconds", training_time)

        # Log final system metrics
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
        mlflow.log_metrics({
            "final/train_loss": train_losses[-1],
            "final/train_accuracy": train_accs[-1],
            "final/val_loss": val_losses[-1],
            "final/val_accuracy": val_accs[-1],
            "final/val_top5_accuracy": val_top5s[-1],
            "best/val_accuracy": best_val_acc,
        })

        # Final validation for artifacts
        _, _, _, all_preds, all_labels = validate(model, val_loader, criterion, device)

        # Save confusion matrix
        cm_path = os.path.join(run_dir, "confusion_matrix.png")
        if log_confusion_matrix(all_preds, all_labels, train_classes, save_path=cm_path):
            print("Confusion matrix created")

        # Save training curves (loss and accuracy plots)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np

            # Loss curve
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(train_losses, label="Train Loss")
            ax.plot(val_losses, label="Val Loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("Training and Validation Loss")
            ax.legend()
            ax.grid(True)
            results_path = os.path.join(run_dir, "results.png")
            plt.savefig(results_path, dpi=150)
            plt.close()
            print("Results chart created")

            # Accuracy curve
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(train_accs, label="Train Accuracy")
            ax.plot(val_accs, label="Val Accuracy")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy (%)")
            ax.set_title("Training and Validation Accuracy")
            ax.legend()
            ax.grid(True)
            acc_path = os.path.join(run_dir, "accuracy.png")
            plt.savefig(acc_path, dpi=150)
            plt.close()
            print("Accuracy chart created")
        except Exception as e:
            print(f"Could not create training curves: {e}")

        # Log all artifacts from run_dir
        # Save class names to file
        class_names_path = os.path.join(run_dir, "classifier_classes.txt")
        with open(class_names_path, "w") as f:
            for i, name in enumerate(train_classes):
                f.write(f"{i}: {name}\n")
        
        artifacts = [
            "weights/best.pth",
            "confusion_matrix.png",
            "results.png",
            "accuracy.png",
            "classifier_classes.txt",
        ]

        for artifact in artifacts:
            path = os.path.join(run_dir, artifact)
            if os.path.exists(path):
                mlflow.log_artifact(path)
                print(f"Logged artifact: {artifact}")

    # После завершения with block - логируем модель и экспортируем в ONNX
    weights_path = "/weights/classifier.pth"
    if os.path.exists(weights_path):
        print(f"\n📦 Модель сохранена: {weights_path}")
        
        # Log classification examples (5 random validation images with predictions)
        try:
            with mlflow.start_run(run_id=run_id):
                log_classification_examples(model, data_dir, train_classes, device, num_examples=5)
        except Exception as e:
            print(f"Could not log classification examples: {e}")
            import traceback
            traceback.print_exc()
        
        # Log as MLflow Model
        try:
            with mlflow.start_run(run_id=run_id):
                log_classifier_model_as_mlflow(weights_path, "classifier")
        except Exception as e:
            print(f"Could not log model: {e}")
            import traceback
            traceback.print_exc()

        # Export to ONNX
        try:
            onnx_path = export_to_onnx(weights_path, "classifier", num_classes, img_size)
            if onnx_path and os.path.exists(onnx_path):
                print(f"📦 ONNX модель: {onnx_path}")
                # Логируем ONNX в тот же run
                with mlflow.start_run(run_id=run_id):
                    mlflow.log_artifact(onnx_path, artifact_path="onnx")
                    print("📦 ONNX модель залогирована в MLflow")
        except Exception as e:
            print(f"Could not export to ONNX: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.2f}%")
    print(f"Experiment ID: {experiment_id}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Обучение классификатора типов самолётов (FGVC Aircraft)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python train_classifier.py --epochs 50
  python train_classifier.py --epochs 100 --batch-size 16
  python train_classifier.py --device cpu
        """,
    )
    parser.add_argument("--data-dir", type=str, default="/ml/data/fgvc_aircraft_imagefolder")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--name", type=str, default="classifier_resnet18")
    parser.add_argument("--device", type=str, default="cuda", help="cuda или cpu")
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        img_size=args.img_size,
        name=args.name,
        device_str=args.device,
    )
