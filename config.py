"""Конфигурация системы мониторинга парковки."""

# Количество парковочных мест
NUM_PARKING_SPOTS = 8

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
# 2. Предобученная YOLOv8 (COCO 80 классов):
#    YOLO_MODEL = "yolov8n.pt"  # nano
#    YOLO_MODEL = "yolov8m.pt"  # medium
#    YOLO_MODEL = "yolov8l.pt"  # large
#
# 3. Скачанная из MLflow (последняя):
#    YOLO_MODEL = "last.pt"
# ==========================================
# YOLO_MODEL = "yolov8x.pt"  # <-- Измените на нужный файл
YOLO_MODEL = "yolov8x.pt"  # <-- Измените на нужный файл

# Тип модели — автоматически определяется по пути:
# - "local" — локальный .pt файл (fine-tuned)
# - "coco" — предобученная COCO модель
# - "mlflow" — из MLflow artifacts
MODEL_TYPE = None  # Определяется автоматически

# Классы для детекции (используется только для COCO моделей):
# None = все классы (для fine-tuned моделей)
# list(range(80)) = все 80 COCO классов
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
