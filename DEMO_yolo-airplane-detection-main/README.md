# YOLO Airplane Detection & Classification

Система детекции и классификации типов самолётов на базе YOLOv8 + ResNet18 с веб-интерфейсом.

## Возможности

- **Детекция самолётов** — YOLOv8-nano, обученная на COCO airplane
- **Модифицированные архитектуры** — 2 упрощённые модели для edge-устройств
- **Классификация типов** — ResNet18, 100 классов (FGVC Aircraft)
- **Веб-интерфейс** — React + TypeScript, загрузка изображений, просмотр результатов
- **История детекций** — хранение в MinIO, фильтрация, поиск
- **MLflow интеграция** — трекинг экспериментов, метрики, реестр моделей
- **ONNX экспорт** — все модели экспортированы для production через ONNX Runtime

## Быстрый старт

### Требования

- Docker + Docker Compose
- NVIDIA GPU (опционально, для обучения)
- 10 GB свободного места

### Запуск

```bash
# Собирать контейнеры
make build

# Запустить все сервисы
make up

# Подождать некоторое время для загрузки бакета в MinIO (10 секунд)

# Загрузить все артефакты в MinIO
make sync-artifacts-to-minio

# Перезапустить Django + React
make restart
```
##### Примечание: MlFlow может не сразу запускаться, нужно дать примерно 10 секунд для синхронизации с бакетом

### Адреса сервисов

| Сервис        | URL                   | Описание              |
| ------------- | --------------------- | --------------------- |
| Frontend      | http://localhost:5173 | Веб-интерфейс (React) |
| Backend API   | http://localhost:8000 | Django REST API       |
| MLflow        | http://localhost:5000 | Трекинг экспериментов |
| MinIO Console | http://localhost:9001 | Хранилище артефактов  |
| MinIO API     | http://localhost:9000 | S3-совместимый API    |

MinIO credentials: `minioadmin` / `minioadmin`

## Архитектура

```
┌─────────────┐     ┌──────────────┐     ┌──────────┐
│   React UI  │────>│  Django API  │────>│  MinIO   │
│  (port 5173)│     │  (port 8000) │     │ (port 9K)│
└─────────────┘     └──────┬───────┘     └──────────┘
                           │
                    ┌──────┴───────┐
                    │  ML Models   │
                    │ YOLO + ResNet│
                    └──────┬───────┘
                           │
                     ┌─────┴─────┐
                     │  MLflow   │
                     │ (port 5K) │
                     └───────────┘
```

### Сервисы

- **Django** — REST API, ONNX Runtime инференс, хранение истории в PostgreSQL/SQLite
- **React (Vite + TypeScript)** — FSD-архитектура, React Query, Ant Design
- **MinIO** — S3-совместимое хранилище изображений и результатов
- **MLflow** — трекинг метрик, параметров, артефактов, реестр моделей

## API

| Метод    | Путь                        | Описание                                  |
| -------- | --------------------------- | ----------------------------------------- |
| `POST`   | `/api/upload/`              | Загрузить изображение, запустить детекцию |
| `POST`   | `/api/detect/`              | Детекция с выбранной моделью              |
| `POST`   | `/api/detect-and-classify/` | Детекция + классификация типа (top-3)     |
| `POST`   | `/api/classify-only/`       | Только классификация изображения          |
| `GET`    | `/api/models/`              | Список доступных моделей                  |
| `GET`    | `/api/mlflow/models/`       | Модели из MLflow registry                 |
| `POST`   | `/api/mlflow/download/`     | Скачать модель из MLflow                  |
| `GET`    | `/api/mlflow/metrics/`      | Метрики экспериментов                     |
| `GET`    | `/api/history/`             | История детекций                          |
| `GET`    | `/api/results/<id>/`        | Результат детекции по ID                  |
| `DELETE` | `/api/results/<id>/delete/` | Удалить результат                         |

### Поддерживаемые модели

| Модель       | Слои | Голов | Размер | mAP50      | FPS (CPU) |
| ------------ | ---- | ----- | ------ | ---------- | --------- |
| `baseline`   | 23   | 3     | ~6 MB  | 0.70       | 89        |
| `mod1`       | 17   | 2     | ~12 MB | 0.70       | 89        |
| `mod2`       | 12   | 1     | ~2 MB  | 0.69       | 60        |
| `classifier` | —    | —     | ~43 MB | 12.78% acc | 188       |

> **Примечание:** Для классификатора используется Accuracy (не mAP50). Top-5 Accuracy: ~42%.

## Датасеты

### Детектор: COCO Airplane

- **Источник**: COCO 2017 (фильтрация по классу "airplane", category_id=5)
- **Размер**: 97 train / 97 val = 194 изображения
- **Формат**: YOLO (txt аннотации)

### Классификатор: FGVC Aircraft

- **Источник**: FGVC Aircraft dataset
- **Классы**: 100 типов (Boeing 737, Airbus A320, F-16A, и т.д.)
- **Размер**: 6,667 train / 3,333 val = 10,000 изображений
- **Формат**: ImageFolder (директории по классам)

### Скачивание датасетов

```bash
# Скачать оба датасета
make download-dataset

# Только COCO
make download-coco

# Только FGVC Aircraft
make download-fgvc
```

## Обучение моделей

### Параметры

| Параметр | Значение по умолчанию |
| -------- | --------------------- |
| EPOCHS   | 100                   |
| BATCH    | 16                    |
| IMGSZ    | 320                   |
| DEVICE   | cpu (gpu при наличии) |

### Команды

```bash
# Baseline YOLOv8n
make train EPOCHS=100

# На CPU
make train-cpu

# Модификации
make train-mod1 EPOCHS=50
make train-mod2 EPOCHS=50

# Классификатор ResNet18
make train-classifier EPOCHS=50

# Все параметры
make train EPOCHS=100 BATCH=8 IMGSZ=320 DEVICE=gpu
```

### Экспорт в ONNX

```bash
make export
```

Все модели автоматически экспортируются при обучении. ONNX-файлы сохраняются в `/weights/`.

## Бенчмарк производительности

```bash
make benchmark
```

### Результаты (CPU)

| Модель       | Размер   | FPS     | ms/кадр |
| ------------ | -------- | ------- | ------- |
| Mod1 (160px) | 11.89 MB | 183-190 | 5.3     |
| Mod1 (320px) | 11.89 MB | 89      | 11.2    |
| Mod2 (320px) | 2.29 MB  | 60      | 16.7    |
| Classifier   | 42.82 MB | 188     | 5.3     |

## Использование ONNX моделей

```python
import onnxruntime as ort
import numpy as np
from PIL import Image

# Детектор
session = ort.InferenceSession("weights/baseline.onnx")
outputs = session.run(None, {"images": input_tensor})

# Классификатор
cls_session = ort.InferenceSession("weights/classifier.onnx")
outputs = cls_session.run(None, {"input": crop_tensor})
```

## Структура проекта

```
yolo-airplane-detection/
├── backend/                    # Django REST API
│   ├── api/                    # Endpoint'ы (upload, detect, classify, history)
│   ├── core/                   # Настройки (settings, MinIO client)
│   ├── detector/               # YOLO-детекция (ONNX Runtime обёртка)
│   └── models/                 # Django модели задач
├── frontend/                   # React (Vite + TypeScript, FSD)
│   └── src/
│       ├── pages/              # Страницы: Detection, History, Dashboard
│       ├── entities/           # Сущности: detection, history, image, metrics
│       ├── features/           # Фичи: upload, run-detection, model selector
│       ├── widgets/            # Виджеты: navbar, uploader, results
│       └── shared/             # Общие: API клиент, типы, конфиги
├── ml/                         # ML-скрипты
│   ├── train_detector.py       # Обучение YOLOv8 (baseline/mod1/mod2)
│   ├── train_classifier.py     # Обучение ResNet18 (FGVC Aircraft)
│   ├── detect_and_classify.py  # Пайплайн детекции + классификации
│   ├── export_onnx.py          # Экспорт моделей в ONNX
│   ├── benchmark_fps.py        # Бенчмарк FPS
│   ├── mlflow_utils.py         # Утилиты MLflow
│   └── data/                   # Датасеты (COCO, FGVC)
├── weights/                    # Веса моделей (.pt, .pth, .onnx)
├── mlruns/                     # MLflow эксперименты и метрики
├── docs/                       # Документация
├── examples/                   # Примеры изображений
├── Makefile                    # Команды управления проектом
├── docker-compose.yml          # Docker Compose (Django, React, MinIO, MLflow)
└── docker-compose.gpu.yml      # GPU overlay (NVIDIA runtime)
```

## Makefile — все команды

### Инфраструктура

| Команда               | Описание                                      |
| --------------------- | --------------------------------------------- |
| `make up`             | Запустить все сервисы (GPU авто-определяется) |
| `make down`           | Остановить все сервисы                        |
| `make restart`        | Перезапустить Django + React                  |
| `make build`          | Собрать/пересобрать сервисы                   |
| `make build-no-cache` | Собрать без кэша                              |
| `make clean`          | Удалить контейнеры и тома                     |
| `make detect-gpu`     | Проверить доступность GPU                     |

### Датасеты

| Команда                 | Описание              |
| ----------------------- | --------------------- |
| `make download-dataset` | Скачать COCO + FGVC   |
| `make download-coco`    | Скачать COCO 2017     |
| `make download-fgvc`    | Скачать FGVC Aircraft |

### Обучение

| Команда                 | Описание                       |
| ----------------------- | ------------------------------ |
| `make train`            | Обучить baseline YOLOv8n       |
| `make train-cpu`        | Обучить baseline на CPU        |
| `make train-mod1`       | Обучить модификацию 1          |
| `make train-mod2`       | Обучить модификацию 2          |
| `make train-classifier` | Обучить классификатор ResNet18 |
| `make export`           | Экспорт моделей в ONNX         |

### MLflow и хранилище

| Команда              | Описание                        |
| -------------------- | ------------------------------- |
| `make mlflow`        | Показать URL MLflow             |
| `make minio`         | Показать URL MinIO              |
| `make backup-mlflow` | Сохранить данные MLflow на хост |

### Логи

| Команда            | Описание                       |
| ------------------ | ------------------------------ |
| `make logs`        | Просмотреть логи всех сервисов |
| `make logs-django` | Логи Django                    |
| `make logs-react`  | Логи React                     |

## Разработка

### Добавить новую модель

1. Создать YAML конфигурацию в `ml/models/yolov8n_custom.yaml`
2. Добавить вариант в `MODEL_CONFIGS` в `ml/train_detector.py`
3. Обучить: `make train-custom EPOCHS=100`

### Frontend разработка

```bash
cd frontend
npm install
npm run dev
```

### Backend разработка

```bash
cd backend
python manage.py migrate
python manage.py runserver
```

## Метрики

### Детектор (YOLO)

- **mAP50** — mean Average Precision при IoU=0.50
- **mAP50-95** — mean Average Precision при IoU=0.50:0.95
- **Precision** — точность детекции
- **Recall** — полнота детекции

### Классификатор (ResNet18)

- **Accuracy** — точность классификации (Top-1)
- **Top-5 Accuracy** — правильный класс в топ-5 предсказаний
- **Loss** — Cross-Entropy loss

## Технологии

| Компонент | Стек                                                |
| --------- | --------------------------------------------------- |
| Backend   | Django 4.x, DRF, ONNX Runtime, boto3 (MinIO)        |
| Frontend  | React 18, TypeScript, Vite, React Query, Ant Design |
| ML        | Ultralytics YOLOv8, PyTorch, ResNet18, scikit-learn |
| Tracking  | MLflow                                              |
| Storage   | MinIO (S3-compatible)                               |
| Infra     | Docker, Docker Compose, Make                        |

## Ссылки

- [COCO Dataset](https://cocodataset.org/)
- [FGVC Aircraft Dataset](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/)
- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- [ONNX Runtime](https://onnxruntime.ai/)
- [MLflow](https://mlflow.org/)
- [Технический отчёт](TECHNICAL_REPORT.md)
