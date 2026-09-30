# Parking Monitor 🚗 — исходный код

Система мониторинга заполненности парковки: детекция **YOLOv8**, разметка зон, видеомониторинг в реальном времени, веб-интерфейс **Gradio** и база данных **PostgreSQL** для истории.

## Как это работает

1. **Детекция** — YOLOv8 находит транспорт на кадре. Модель задаётся в `config.py` (`YOLO_MODEL`), веса лежат в `models/`.
2. **Зоны** — каждое место задаётся прямоугольником (или полигоном) и точкой номера. Хранятся в JSON.
3. **Занятость** — место занято, если центр bbox объекта попадает в зону или bbox пересекает границы зоны (`zones.py:update_states_by_center()`). Плюс временное сглаживание.
4. **Визуализация** — зоны зелёные (свободно) / красные (занято), счётчик свободных/занятых.

## Веб-интерфейс (Gradio)

```powershell
python main.py
```

| Вкладка | Назначение |
|---------|-----------|
| **📸 Обработка** | Загрузить фото — разметка зон и счёт; результат пишется в БД |
| **📐 Разметка зон** | Нарисовать зоны: 5 кликов = 4 угла + номер места |
| **🖼️ Кадр** | Извлечь кадр из видео для разметки |
| **🎬 OpenCV** | Запустить видеомониторинг в окне OpenCV (галочка «Писать в БД») |
| **📋 Пульт оператора** | Состояние мест и журнал посещений (из PostgreSQL, автообновление) |

## Видеомониторинг

Видео обрабатывается в нативном разрешении, зоны хранятся пофайлово в `zones/<имя_видео>.json`.

```powershell
# 1. Разметка зон под видео (ESC — выход, зоны сохраняются)
python setup_zones_cv.py --video "image/Video/1790685044033-...mp4"

# 2. Мониторинг (+ запись снимков в БД, source = имя видео)
python run_cv_video.py --video "image/Video/1790685044033-...mp4" --db

# опции: --zones <файл> --scale 0.8 --model models/yolov8m.pt --loop --db-interval 2
```

`run_camera.py` — веб-камера, `run_parking.py` — видео с трекингом. Они используют зоны из `parking_zones.json`.

## База данных PostgreSQL

История заполняемости и сессии стоянок (приезд/уезд). Подробно — `db/README.md`.

```powershell
python -m db.init_db     # применить схему + загрузить зоны
python -m db.cli         # состояние мест и журнал в терминале
.\db\psql.cmd            # интерактивный psql
```

Таблицы: `parking_zones`, `occupancy_snapshots`, `zone_occupancy`, `parking_sessions`.
Вьюхи: `v_parking_status` (состояние мест), `v_sessions_journal` (журнал), `v_operator_board`, `v_occupancy_by_hour`, `v_occupancy_by_weekday`, `v_zone_stats`.

## Модели

Веса в `models/` (в git не хранятся): `yolov8n/m/x.pt`, дообученная `yolov8m_parking_best.pt`. По умолчанию — `models/yolov8x.pt`. Описание — `models/README.md`.

## Дообучение

```powershell
python finetune.py --epochs 30 --model models/yolov8m.pt --imgsz 320 --batch 8
```

Датасет парковок (Kaggle), метрики и артефакты — в MLflow, экспорт в ONNX.

## Установка

```powershell
pip install -r requirements.txt
```

## Файлы проекта

| Файл | Назначение |
|------|-----------|
| `main.py` | Gradio UI: обработка, разметка, кадр, OpenCV, пульт оператора |
| `config.py` | Модель, пороги, `VIDEO_FILE`, число мест |
| `detector.py` | Обёртка YOLO: `detect()`, `detect_with_tracking()` |
| `zones.py` | `ParkingZones`: загрузка/сохранение, проверка занятости, отрисовка, загрузчики зон |
| `parking_zones.json` | Зоны для вкладки «Обработка» / веб-камеры |
| `zones/` | Зоны, отдельные для каждого видео |
| `run_cv_video.py` | Видеомониторинг (OpenCV) + запись в БД |
| `run_camera.py` / `run_parking.py` | Камера / видео с трекингом |
| `setup_zones.py` | Редактор зон на Tkinter |
| `setup_zones_cv.py` | Редактор зон на OpenCV (по видео) |
| `extract_frame.py` | Извлечь кадр из видео |
| `finetune.py` | Дообучение YOLOv8 (MLflow, ONNX) |
| `db/` | Модуль PostgreSQL (схема, репозиторий, CLI) |
| `models/` | Веса моделей |
| `requirements.txt` | Зависимости |

## Технологии

Python 3.12 · Ultralytics YOLOv8 · Gradio · OpenCV · NumPy · Pillow · PostgreSQL · SQLAlchemy · pandas · MLflow

## Автор

👋 **Игорь Половников** — студент МГТУ им. Н.Э. Баумана. Разрабатываю полноценные системы: от прошивки микроконтроллеров и компьютерного зрения до веб-интерфейсов. Python, Rust, C/C++, ESP32, Django, React.
