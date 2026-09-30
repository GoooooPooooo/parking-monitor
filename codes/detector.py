"""Обёртка над YOLO для детекции и трекинга всех объектов."""

from ultralytics import YOLO
import numpy as np
import cv2
import os
import config


# Все классы COCO (80 классов)
COCO_CLASSES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
    5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
    10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird',
    15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
    20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
    25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
    30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
    35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
    40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon',
    45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
    50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut',
    55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
    60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse',
    65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven',
    70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock',
    75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}


def resolve_model_path(model_name: str) -> str:
    """Найти путь к модели: локальный файл → MLflow → скачивание."""
    # 1. Проверяем локальный файл
    if os.path.exists(model_name):
        print(f"✅ Найдена локальная модель: {os.path.abspath(model_name)}")
        return model_name

    # 2. Проверяем в директории проекта
    project_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_name)
    if os.path.exists(project_path):
        print(f"✅ Найдена модель в проекте: {project_path}")
        return project_path

    # 3. Проверяем MLflow artifacts
    mlflow_path = os.path.join("mlruns", model_name)
    if os.path.exists(mlflow_path):
        print(f"✅ Найдена модель в MLflow: {mlflow_path}")
        return mlflow_path

    # 4. Ищем в MLflow рекурсивно
    for root, dirs, files in os.walk("mlruns"):
        if model_name in files:
            found = os.path.join(root, model_name)
            print(f"✅ Найдена модель в MLflow: {found}")
            return found

    # 5. Не найден — YOLO скачает предобученную (yolov8*.pt)
    print(f"⚠️ Модель '{model_name}' не найдена локально, будет скачана предобученная")
    return model_name


def get_model_classes(model: YOLO) -> list:
    """Получить список классов модели."""
    if hasattr(model, 'names'):
        return model.names
    return COCO_CLASSES


class CarDetector:
    """Детектор ВСЕХ объектов на базе YOLOv8."""

    def __init__(self):
        # Определяем путь к модели
        model_path = resolve_model_path(config.YOLO_MODEL)
        self.model = YOLO(model_path)

        # Получаем классы модели
        self.model_classes = get_model_classes(self.model)
        self.is_finetuned = len(self.model_classes) < 80  # Fine-tuned модель

        if self.is_finetuned:
            print(f"✅ Fine-tuned модель: {len(self.model_classes)} классов")
            for cls_id, cls_name in self.model_classes.items():
                print(f"   [{cls_id}] {cls_name}")
        else:
            print(f"✅ Предобученная COCO модель: 80 классов")

    def detect(self, image):
        """
        Детекция ВСЕХ объектов на изображении.

        Args:
            image: numpy array (BGR или RGB)

        Returns:
            Список всех детекций: [(x1, y1, x2, y2, confidence, class_id), ...]
        """
        results = self.model(image, conf=config.CONFIDENCE_THRESHOLD, verbose=False)

        detections = []
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                if config.DETECT_CLASSES is not None and cls not in config.DETECT_CLASSES:
                    continue
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append((x1, y1, x2, y2, conf, cls))

        # Отладка — показать ВСЕ найденные объекты
        if config.DEBUG_MODE:
            print(f"\n🔍 YOLO нашёл {len(detections)} объектов (порог={config.CONFIDENCE_THRESHOLD}):")
            for x1, y1, x2, y2, conf, cls in detections:
                name = self.model_classes.get(cls, f'class_{cls}')
                print(f"  [{cls:2d}] {name:20s} conf={conf:.3f}  bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")
            print()

        return detections

    def get_class_name(self, cls_id):
        """Получить название класса по ID."""
        return self.model_classes.get(cls_id, f'class_{cls_id}')

    def detect_with_tracking(self, image, persist=True):
        """
        Детекция с трекингом объектов.

        Args:
            image: numpy array (BGR или RGB)
            persist: сохранять ID между кадрами

        Returns:
            Список детекций с track_id: [(x1, y1, x2, y2, conf, cls, track_id), ...]
            Каждый объект содержит:
            - x1, y1, x2, y2: bbox
            - conf: уверенность
            - cls: класс объекта
            - track_id: ID трека (или None если нет трекинга)
        """
        results = self.model.track(image, persist=persist, verbose=False)

        detections = []
        for result in results:
            if result.boxes is None or result.boxes.id is None:
                for box in result.boxes:
                    cls = int(box.cls[0])
                    if config.DETECT_CLASSES is not None and cls not in config.DETECT_CLASSES:
                        continue
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detections.append((x1, y1, x2, y2, conf, cls, None))
                continue

            ids = result.boxes.id.cpu().numpy().astype(int)
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.int().cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()

            for i in range(len(ids)):
                x1, y1, x2, y2 = boxes[i]
                conf = confs[i]
                cls = int(classes[i])
                if config.DETECT_CLASSES is not None and cls not in config.DETECT_CLASSES:
                    continue
                track_id = ids[i]
                detections.append((x1, y1, x2, y2, conf, cls, track_id))

        if config.DEBUG_MODE:
            print(f"\n🔍 YOLO нашёл {len(detections)} объектов:")
            for x1, y1, x2, y2, conf, cls, tid in detections:
                name = self.model_classes.get(cls, f'class_{cls}')
                print(f"  [{cls:2d}] {name:20s} conf={conf:.3f} id={tid} bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")
            print()

        return detections

    def get_object_center(self, bbox):
        """Получить центр объекта по bbox."""
        x1, y1, x2, y2 = bbox[:4]
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        return cx, cy

    def is_point_in_polygon(self, point, polygon):
        """
        Проверить, находится ли точка внутри полигона.

        Args:
            point: (x, y)
            polygon: numpy массив точек формы (N, 1, 2)

        Returns:
            True если точка внутри полигона
        """
        return cv2.pointPolygonTest(polygon, point, False) >= 0
