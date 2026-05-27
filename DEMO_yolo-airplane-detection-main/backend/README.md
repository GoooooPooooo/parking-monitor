# Backend — Django API для детекции самолётов

## Структура проекта

```
backend/
├── api/               # Django приложение API
│   ├── views.py       # ViewSet для endpoint'ов
│   ├── serializers.py # DRF сериализаторы
│   └── urls.py        # Маршруты API
├── core/              # Основные настройки проекта
│   ├── settings.py    # Настройки Django
│   ├── urls.py        # Корневой URL конфиг
│   ├── storage.py     # MinIO клиент
│   └── wsgi.py        # WSGI конфиг
├── detector/          # Логика YOLO детекции
│   ├── detector.py    # Обёртка для YOLO модели
│   ├── classifier.py  # Классификатор самолётов
│   └── image_utils.py # Утилиты обработки изображений
├── models/            # Django модели
│   └── models.py      # DetectionTask, DetectionResult
└── manage.py
```

## API Endpoints

| Метод | Путь | Описание |
|-------|------|----------|
| GET | `/api/models/` | Список доступных моделей |
| POST | `/api/detect/` | Запуск детекции на изображении |
| POST | `/api/upload/` | Загрузка изображения |
| GET | `/api/results/<id>/` | Получение результатов |

## Запуск

```bash
# Через Docker Compose (из корня проекта)
docker compose up -d

# Миграции
docker compose exec django python manage.py migrate

# Логи
docker compose logs -f django
```

## Зависимости

Все зависимости в `requirements.txt` в корне проекта:
- Django 4.2+
- Django REST Framework
- boto3 (MinIO)
- opencv-python-headless
- ultralytics (YOLO)
- onnxruntime
- mlflow
