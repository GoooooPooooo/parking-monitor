# Управление MLflow моделями в репозитории

Этот документ описывает процесс добавления обученных моделей MLflow в git репозиторий.

## Быстрый старт

### Просмотр доступных моделей

```bash
make list-mlflow-runs
```

Показывает все запуски по всем экспериментам.

### Добавление модели в репозиторий

```bash
make add-mlflow-model RUN_ID=d86dbee65f9b4e70b2c98801d0c74a49
```

Эта команда:

1. **Автоматически определяет** эксперимент по `RUN_ID` (не нужно указывать `EXPERIMENT_ID`)
2. Скачивает все артефакты модели из MinIO через Python/boto3 (веса, ONNX, графики, примеры)
3. Обновляет `.gitignore` для включения этой модели
4. Добавляет файлы в git staging area

### Завершение процесса

После выполнения команды создайте коммит:

```bash
git commit -m "feat: добавить модель yolov8n_mod1 (d86dbee6)"
git push
```

### Синхронизация всех локальных артефактов в MinIO

```bash
make sync-artifacts-to-minio
```

Загружает все локальные артефакты из `mlruns/` в MinIO, исправляет `artifact_uri` в `meta.yaml` и автоматически перезапускает MLflow (если контейнер запущен).

## Что включается в репозиторий

### Детекторы (YOLOv8)

- **Метрики обучения**: box_loss, cls_loss, dfl_loss, mAP50, mAP50-95, precision, recall по эпохам
- **Параметры модели**: architecture, batch, epochs, imgsz, device, num_classes и т.д.
- **Артефакты**:
  - `best.pt` / `last.pt` — веса модели
  - `onnx/yolov8n_mod1.onnx` / `yolov8n_mod2.onnx` — ONNX экспорт (~12MB)
  - `confusion_matrix.png`, `confusion_matrix_normalized.png`, `results.png`, `labels.jpg` — графики
  - `detection_examples/` — примеры детекций (5 изображений)
  - `model_yolov8n_mod*/` — MLflow модель (python_model.pkl, MLmodel, requirements.txt)

### Классификаторы (ResNet18)

- **Метрики**: accuracy, loss по эпохам
- **Артефакты**:
  - `best.pth` — веса модели (~43MB)
  - `onnx/classifier.onnx` — ONNX экспорт (~44MB)
  - `accuracy.png`, `confusion_matrix.png`, `results.png` — графики
  - `classification_examples/` — примеры классификации (5 изображений)
  - `model_classifier/` — MLflow модель

## Требования

- Docker с запущенным MinIO контейнером
- Python 3 с установленным `boto3` (есть в `requirements.txt`)
- Доступ к MinIO на `localhost:9000` с credentials `minioadmin:minioadmin`

## Структура директорий

```
mlruns/
└── 382940614344375127/           # Experiment: yolov8_airplane_detection
    ├── meta.yaml                 # Метаданные эксперимента (artifact_location: s3://mlflow/...)
    ├── 0839949166a44692890c68e3bf592de7/  # yolov8n_mod2
    ├── d86dbee65f9b4e70b2c98801d0c74a49/  # yolov8n_mod1
    ├── 939a5278899d48f6863b3145748b76e5/  # yolov8n_mod1
    ├── 9eb2c2f4d7224a24ba322922172cee32/  # yolov8n_mod2
    ├── d05990654c1e48a0968d30b42416166c/  # yolov8n_mod1
    ├── 30ca6a8e7b2c4da5a79b76fb2c96e851/  # classifier_resnet18
    ├── 47bec581297e4c2da4128660b43dd1d2/  # classifier_resnet18
    └── 47e7a82258f14e73afaffeff8f41033f/  # classifier_resnet18
```

## Скрипты

### `scripts/download_mlflow_artifacts.py`

Скачивает артефакты конкретного запуска из MinIO. Используется внутри `make add-mlflow-model`.

```bash
python3 scripts/download_mlflow_artifacts.py <experiment_id> <run_id> <artifacts_dir>
```

### `scripts/sync_mlflow_artifacts.py`

Утилита для синхронизации артефактов с MinIO (upload/download/fix-uri).

```bash
# Загрузить все локальные артефакты в MinIO
python3 scripts/sync_mlflow_artifacts.py upload mlruns --fix-uri

# Скачать артефакты конкретного запуска
python3 scripts/sync_mlflow_artifacts.py download <exp_id> <run_id> <artifacts_dir>
```

## Примечания

- По умолчанию все `mlruns/` игнорируются в `.gitignore`
- Только явно добавленные модели через `make add-mlflow-model` попадут в репозиторий
- Артефакты хранятся в MinIO и скачиваются при добавлении модели
- `EXPERIMENT_ID` определяется автоматически — не нужно указывать вручную
- `sync-artifacts-to-minio` проверяет, запущен ли MLflow контейнер, и перезапускает его только если он активен

## Troubleshooting

### MinIO не доступен

Убедитесь, что MinIO контейнер запущен:

```bash
docker ps | grep minio
```

Если не запущен:

```bash
docker-compose up -d minio
```

### Модель не найдена

Проверьте список доступных моделей:

```bash
make list-mlflow-runs
```

### Ошибка прав доступа (Permission denied)

Файлы в `mlruns/` могут принадлежать root, если были созданы Docker-контейнером. Исправьте:

```bash
sudo chown -R $USER:$USER mlruns/
```

### `boto3` не найден

Установите зависимости:

```bash
pip3 install boto3
# или
pip3 install -r requirements.txt
```

> **Важно:** Не запускайте `make sync-artifacts-to-minio` через `sudo` — Python от root не видит пакеты из user site-packages (`~/.local/lib/`).
