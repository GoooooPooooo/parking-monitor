"""Конфигурация системы мониторинга парковки."""

# Количество парковочных мест
NUM_PARKING_SPOTS = 8

# Видео по умолчанию (для вкладки OpenCV и run_cv_video.py) — путь относительно codes/
VIDEO_FILE = "image/Video/1790685044033-01a0ed25-3f0d-74be-b025-4a69044c279a.mp4"

# Порог уверенности YOLO (минимальный — показываем всё)
CONFIDENCE_THRESHOLD = 0.15

# Порог покрытия зоны bbox'ом — место занято если bbox покрывает >30% зоны
COVERAGE_THRESHOLD = 0.3  # 30%

# Сглаживание: сколько последних кадров учитывать для стабильности
SMOOTHING_FRAMES = 3

# ==========================================
# МОДЕЛЬ YOLO — 3 варианта:
# ==========================================
# 1. Локальный файл (рекомендуется):
#    YOLO_MODEL = "best.pt"  # Файл в текущей директории
#    YOLO_MODEL = "C:/parking-monitor/best.pt"  # Абсолютный путь
#
# 2. Предобученная YOLOv8 (COCO 80 классов) — папка models/:
#    YOLO_MODEL = "models/yolov8n.pt"  # nano (быстрая)
#    YOLO_MODEL = "models/yolov8m.pt"  # medium (баланс)
#    YOLO_MODEL = "models/yolov8x.pt"  # x-large (точная)
#
# 3. Скачанная из MLflow (последняя):
#    YOLO_MODEL = "last.pt"
# ==========================================
# YOLO_MODEL = "models/yolov8x.pt"  # <-- Измените на нужный файл
YOLO_MODEL = "models/yolov8x.pt"  # <-- Измените на нужный файл

# Тип модели — автоматически определяется по пути:
# - "local" — локальный .pt файл (fine-tuned)
# - "coco" — предобученная COCO модель
# - "mlflow" — из MLflow artifacts
MODEL_TYPE = None  # Определяется автоматически

# Классы для детекции (фильтр детекций по class_id):
# None = все классы из модели
# [0, 2, 5, 7] = person, car, bus, truck (для COCO)
DETECT_CLASSES = None  # None = все классы из модели

# Режим отладки — показывать все детекции в консоли
DEBUG_MODE = True


def get_model_type(model_path: str) -> str:
    """Определить тип модели по пути."""
    import os
    # Локальный файл
    if os.path.exists(model_path):
        return "local"
    # Предобученная YOLOv8
    if model_path.startswith("yolov8") and model_path.endswith(".pt"):
        return "coco"
    # MLflow (последняя)
    if model_path in ("last.pt", "best.pt") and not os.path.exists(model_path):
        return "mlflow"
    return "local"  # По умолчанию — локальный файл


# Автоматически определяем тип
MODEL_TYPE = get_model_type(YOLO_MODEL)
