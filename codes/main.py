"""
Parking Monitor — Мониторинг парковки по изображению + разметка зон в браузере.
Запуск: python main.py
"""

import gradio as gr
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import json
import cv2
import os
import time

from detector import CarDetector, COCO_CLASSES
from zones import ParkingZones
import config

# Глобальные переменные
detector = None
parking_zones = None
current_image = None
edit_points = []
edit_zones = []
edit_mode = False
edit_shape_type = "rect"  # "rect" или "poly"


def init_system():
    """Инициализация детектора и зон."""
    global detector, parking_zones

    print("Загрузка YOLO модели...")
    detector = CarDetector()
    print("✅ Модель загружена")

    print("Загрузка зон парковки...")
    parking_zones = ParkingZones("parking_zones.json")
    print(f"✅ Загружено {len(parking_zones.zones)} зон")

    free, occupied, total = parking_zones.get_counts()
    return f"✅ Система готова | Свободно: {free} | Занято: {occupied}"


def _to_pil(image_data):
    """Безопасная конвертация в PIL Image."""
    if image_data is None:
        return None
    if isinstance(image_data, np.ndarray):
        return Image.fromarray(image_data)
    if isinstance(image_data, Image.Image):
        return image_data.copy()
    return None


def process_image(input_image):
    """Обработка загруженного изображения - метод центра объекта."""
    global detector, parking_zones

    if detector is None or parking_zones is None:
        init_system()

    if input_image is None:
        return None, "Нет изображения"

    if detector is None:
        return input_image, "Ошибка: детектор не загружен"

    print(f"📥 Тип input: {type(input_image)}")

    # Gradio возвращает numpy array напрямую или PIL Image
    try:
        if hasattr(input_image, 'astype'):
            frame = input_image
        elif isinstance(input_image, Image.Image):
            frame = np.array(input_image)
        else:
            frame = np.array(input_image)
    except Exception as e:
        return None, f"Ошибка конвертации: {e}"

    if frame is None or frame.size == 0:
        return None, "Пустое изображение"

    print(f"📷 Форма: {frame.shape}, dtype: {frame.dtype}")

    # Убираем альфа-канал если есть
    if len(frame.shape) == 3 and frame.shape[2] == 4:
        frame = frame[:, :, :3]

    detections = detector.detect(frame)
    print(f"🔍 Найдено объектов: {len(detections)}")

    parking_zones.update_states_by_center(detections)

    # Рисуем зоны
    image_with_zones = parking_zones.draw_zones(frame)

    free, occupied, total = parking_zones.get_counts()
    log_snapshot_to_db(source="main")
    status = f"🅿️ Свободно: {free} | Занято: {occupied} | Всего: {total}"

    return image_with_zones, status


def process_video(input_video_path, progress=gr.Progress()):
    """Обработка видеофайла - метод центра объекта (как на YouTube)."""
    import cvzone

    global detector, parking_zones

    if detector is None or parking_zones is None:
        init_system()

    if input_video_path is None:
        return None, "Нет видеофайла"

    if not os.path.exists(input_video_path):
        return None, "Файл не найден"

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        return None, "Не удалось открыть видеофайл"

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\n📹 Видео: {width}x{height}, {fps:.1f} FPS, {total_frames} кадров")

    output_path = "output_parking.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    skip_frames = 3

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % skip_frames != 0:
                continue

            frame = cv2.resize(frame, (1020, 500))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            detections = detector.detect_with_tracking(frame_rgb, persist=True)

            parking_zones.update_states_by_center(detections)

            occupied_zones = 0
            for zone_idx, zone in enumerate(parking_zones.zones):
                pts = parking_zones._get_zone_points(zone)
                if pts is None:
                    continue
                pts_np = np.array(pts, np.int32).reshape((-1, 1, 2))

                color = (0, 0, 255) if parking_zones.zone_states[zone_idx] else (0, 255, 0)
                cv2.polylines(frame, [pts_np], isClosed=True, color=color, thickness=2)

                if parking_zones.zone_states[zone_idx]:
                    occupied_zones += 1

            for det in detections:
                x1, y1, x2, y2 = det[:4]
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                cv2.circle(frame, (cx, cy), 4, (255, 0, 255), -1)

            total_zones = len(parking_zones.zones)
            free_zones = total_zones - occupied_zones

            print(f"Free zones: {free_zones}")
            cvzone.putTextRect(frame, f'FREEZONE:{free_zones}', (30, 40), 2, 2)
            cvzone.putTextRect(frame, f'OCC:{occupied_zones}', (30, 140), 2, 2)

            annotated_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(annotated_bgr)

            if frame_count % (skip_frames * 10) == 0 or frame_count >= total_frames:
                pct = frame_count / max(total_frames, 1)
                progress(pct, desc=f"Кадр {frame_count}/{total_frames}")

    finally:
        cap.release()
        out.release()

    final_status = (
        f"✅ Обработано {frame_count} кадров\n"
        f"🅿️ Итог: Свободно: {free_zones} | Занято: {occupied_zones} | Всего: {total_zones}\n"
        f"📁 Результат: {output_path}"
    )

    print(f"\n{final_status}")
    return output_path, final_status


# ==================== РЕЖИМ РАЗМЕТКИ ====================

def start_edit_mode(image):
    """Начать режим разметки."""
    global current_image, edit_points, edit_zones, edit_mode, edit_shape_type

    if image is None:
        return "Сначала загрузите изображение", "", None, "Прямоугольник"

    current_image = image
    edit_points = []
    edit_zones = []
    edit_mode = True
    edit_shape_type = "rect"

    # Конвертируем для отображения
    display_img = _to_pil(image)

    return (
        "Кликайте 4 угла парковочного места → 5-й клик = номер места",
        "Этап: места | Зон: 0 | Точек: 0/5",
        display_img,
        "Прямоугольник"
    )


def on_image_click(evt: gr.SelectData):
    """Обработка клика на изображении."""
    global edit_points, edit_zones, current_image, edit_shape_type

    if not edit_mode or current_image is None:
        return "Не в режиме разметки", "", None, edit_shape_type

    x, y = int(evt.index[0]), int(evt.index[1])
    edit_points.append((x, y))

    # ОБЯЗАТЕЛЬНО конвертируем в PIL
    img_copy = _to_pil(current_image)
    if img_copy is None:
        return "Ошибка изображения", "", None, edit_shape_type

    draw = ImageDraw.Draw(img_copy)

    # Рисуем созданные зоны
    for zone_idx, zone_data in enumerate(edit_zones):
        _draw_edit_zone(draw, zone_data, zone_idx + 1)

    num_point = len(edit_points)
    shape_label = "Прямоугольник" if edit_shape_type == "rect" else "Параллелограмм"
    info = f"Клик: ({x}, {y})"

    if num_point <= 4:
        r = 5
        draw.ellipse([x - r, y - r, x + r, y + r], fill="yellow", outline="black")
        info += f" | Угол {num_point}/4"
        # Соединяем точки линией для параллелограмма
        if edit_shape_type == "poly" and len(edit_points) > 1:
            for i in range(len(edit_points) - 1):
                p1 = edit_points[i]
                p2 = edit_points[i + 1]
                draw.line([p1[0], p1[1], p2[0], p2[1]], fill="yellow", width=2)
    elif num_point == 5:
        r = 8
        draw.ellipse([x - r, y - r, x + r, y + r], fill="cyan", outline="black")
        info += " | Номер места"

        corners = edit_points[:4]
        label_x, label_y = edit_points[4]

        if edit_shape_type == "rect":
            xs = [p[0] for p in corners]
            ys = [p[1] for p in corners]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            zone = {"type": "rect", "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                     "label_x": label_x, "label_y": label_y}
        else:
            # Параллелограмм — 4 точки в порядке клика
            zone = {"type": "poly", "points": [list(p) for p in corners],
                     "label_x": label_x, "label_y": label_y}

        edit_zones.append(zone)
        info += f" | ✅ Зона {len(edit_zones)} создана"
        edit_points = []

    pts_left = 5 - len(edit_points)
    status_text = f"Этап: места | Зон: {len(edit_zones)}"
    if edit_points:
        status_text += f" | Точек: {len(edit_points)}/5 (осталось: {pts_left})"

    return info, status_text, img_copy, shape_label


def _draw_edit_zone(draw, zone, zone_num):
    """Нарисовать зону в режиме разметки."""
    if zone["type"] == "rect":
        x1, y1, x2, y2 = zone["x1"], zone["y1"], zone["x2"], zone["y2"]
        draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    else:
        pts = [tuple(p) for p in zone["points"]]
        draw.polygon(pts, outline="green", width=2)
        cx = sum(p[0] for p in pts) // 4
        cy = sum(p[1] for p in pts) // 4

    r = 10
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill="cyan")
    draw.text((cx - 5, cy - 7), str(zone_num), fill="black")


def toggle_shape_type():
    """Переключить тип формы."""
    global edit_shape_type
    edit_shape_type = "poly" if edit_shape_type == "rect" else "rect"
    label = "Параллелограмм" if edit_shape_type == "poly" else "Прямоугольник"
    btn_text = "🔄 Прямоугольник" if edit_shape_type == "poly" else "🔄 Параллелограмм"
    return label, btn_text


def save_zones():
    """Сохранить размеченные зоны."""
    global edit_zones, parking_zones

    if not edit_zones:
        return "Нет зон для сохранения"

    filepath = "parking_zones.json"
    data = {'zones': edit_zones}
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

    parking_zones = ParkingZones(filepath)

    return f"✅ Сохранено {len(edit_zones)} зон в {filepath}"


def cancel_edit():
    """Выйти из режима разметки."""
    global edit_mode, edit_points, edit_zones, edit_shape_type
    edit_mode = False
    edit_points = []
    edit_zones = []
    edit_shape_type = "rect"
    return "Режим разметки отключён", "", "Прямоугольник"


# ==================== БАЗА ДАННЫХ ====================

def _db():
    from db import repository
    return repository


def log_snapshot_to_db(source="main"):
    """Записать текущее состояние зон в БД. Ошибки БД не должны ломать приложение."""
    try:
        repository = _db()
        if parking_zones is None:
            return None
        db_zones = repository.get_zones()
        if not db_zones:
            return None
        states = [
            {"zone_id": db_zones[i]["id"], "is_occupied": bool(occupied)}
            for i, occupied in enumerate(parking_zones.zone_states)
            if i < len(db_zones)
        ]
        snapshot_id = repository.log_snapshot(states, source=source, model=config.YOLO_MODEL)
        print(f"🗄️ Снимок записан в БД: id={snapshot_id}")
        return snapshot_id
    except Exception as e:
        print(f"⚠️ БД недоступна, снимок не записан: {e}")
        return None


def db_init():
    """Применить схему и загрузить зоны в БД."""
    try:
        repository = _db()
        repository.init_schema()
        n = repository.sync_zones_from_json()
        return f"✅ БД готова. Схема применена, зон загружено: {n}"
    except Exception as e:
        return f"❌ Ошибка БД: {e}"


def operator_refresh(source=None):
    """Данные для пульта оператора: сводка, состояние мест, журнал."""
    try:
        repository = _db()
        sources = repository.list_sources() or ["main"]
        if source not in sources:
            source = repository.latest_source() or sources[0]

        summary = repository.board_summary(source)
        board = pd.DataFrame(repository.operator_board(source))
        journal = pd.DataFrame(repository.sessions_journal(20, source))

        text = (f"### Объект: `{source}`\n"
                f"## 🟢 Свободно: {summary['free']} из {summary['total']}   "
                f"|   🔴 Занято: {summary['occupied']}")
        if board.empty:
            text = "## Данных пока нет — разметьте зоны и обработайте изображение/видео."
        return text, board, journal, gr.update(choices=sources, value=source)
    except Exception as e:
        return f"### ❌ Ошибка БД: {e}", pd.DataFrame(), pd.DataFrame(), gr.update()


# ==================== ИНТЕРФЕЙС ====================

def create_interface():
    """Создать Gradio интерфейс."""

    with gr.Blocks(title="Parking Monitor") as app:
        gr.Markdown("# 🚗 Мониторинг парковки")
        gr.Markdown("Система отслеживания заполненности парковочных мест на базе YOLO")

        with gr.Tabs():
            # Вкладка 1: Обработка
            with gr.Tab("📸 Обработка"):
                status_box = gr.Textbox(
                    label="Статус системы",
                    value="Нажмите 'Инициализация системы'",
                    interactive=False
                )
                init_btn = gr.Button("⚙️ Инициализация системы", variant="primary")

                gr.Markdown("---")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Исходное изображение")
                        image_input = gr.Image(label="Загрузите фото парковки")

                    with gr.Column():
                        gr.Markdown("### Результат обработки")
                        output_image = gr.Image(label="С детекцией и зонами")
                        status_text = gr.Textbox(label="Статус", value="Ожидание...", interactive=False)

                process_btn = gr.Button("📸 Обработать изображение", variant="primary", size="lg")

                init_btn.click(fn=init_system, outputs=status_box)
                process_btn.click(
                    fn=process_image,
                    inputs=image_input,
                    outputs=[output_image, status_text]
                )

            # Вкладка 2: Разметка зон
            with gr.Tab("📐 Разметка зон"):
                gr.Markdown("### Интерактивная разметка парковочных зон")
                gr.Markdown("**Шаг 1:** Загрузите фото → 'Начать разметку'\n"
                            "**Шаг 2:** 5 кликов на каждое место (4 угла + номер) → Сохранить\n"
                            "**Совет:** Используйте '🔄 Параллелограмм' для парковок-ёлочек")

                with gr.Row():
                    edit_image_input = gr.Image(label="Фото для разметки")
                    edit_image_output = gr.Image(label="Размеченные зоны", interactive=False)

                edit_status = gr.Textbox(label="Статус разметки", value="Загрузите фото и нажмите 'Начать разметку'", interactive=False)
                edit_zone_count = gr.Textbox(label="Прогресс", value="", interactive=False)

                with gr.Row():
                    shape_type_display = gr.Textbox(label="Тип формы", value="Прямоугольник", interactive=False)
                    toggle_shape_btn = gr.Button("🔄 Параллелограмм", variant="secondary")

                with gr.Row():
                    start_edit_btn = gr.Button("▶️ Начать разметку", variant="primary")
                    save_btn = gr.Button("💾 Сохранить зоны", variant="primary")
                    cancel_btn = gr.Button("❌ Отменить", variant="stop")

                toggle_shape_btn.click(
                    fn=toggle_shape_type,
                    outputs=[shape_type_display, toggle_shape_btn]
                )

                start_edit_btn.click(
                    fn=start_edit_mode,
                    inputs=edit_image_input,
                    outputs=[edit_status, edit_zone_count, edit_image_output, shape_type_display]
                )

                edit_image_input.select(
                    fn=on_image_click,
                    outputs=[edit_status, edit_zone_count, edit_image_output, shape_type_display]
                )

                save_btn.click(fn=save_zones, outputs=edit_status)
                cancel_btn.click(fn=cancel_edit, outputs=[edit_status, edit_zone_count, shape_type_display])

            # Вкладка 3: Извлечь кадр из видео
            with gr.Tab("🖼️ Кадр"):
                gr.Markdown("### Извлечь кадр из видео для разметки")
                
                extract_btn = gr.Button("📷 Извлечь кадр", variant="primary")
                extract_status = gr.Textbox(label="Статус", value="Нажмите кнопку для извлечения")
                
                def extract_frame():
                    import subprocess
                    result = subprocess.run(["python", "extract_frame.py"], capture_output=True, text=True)
                    return result.stdout.strip()
                
                extract_btn.click(fn=extract_frame, outputs=extract_status)

            # Вкладка 4: OpenCV видео + мышь
            with gr.Tab("🎬 OpenCV"):
                gr.Markdown("### Видеомониторинг в окне OpenCV")
                gr.Markdown("Зоны берутся из `zones/<имя_видео>.json` "
                            "(если нет — из `parking_zones.json`).\n"
                            "**Управление:** ESC = выход")

                cv_video_input = gr.Textbox(label="Путь к видео", value=config.VIDEO_FILE)
                cv_db_check = gr.Checkbox(label="Писать снимки в БД (историю)", value=True)
                run_cv_btn = gr.Button("🎬 Запустить видео", variant="primary")
                cv_status = gr.Textbox(label="Статус", interactive=False)

                def launch_cv_video(video_path, write_db):
                    import subprocess
                    import os

                    if not video_path or not os.path.exists(video_path):
                        return f"⚠️ Видео не найдено: {video_path}"

                    cmd = ["python", "run_cv_video.py", "--video", video_path]
                    if write_db:
                        cmd.append("--db")
                    subprocess.Popen(cmd)
                    return f"✅ Окно OpenCV запущено: {video_path}" + (" (запись в БД)" if write_db else "")

                run_cv_btn.click(fn=launch_cv_video, inputs=[cv_video_input, cv_db_check], outputs=cv_status)

            # Вкладка 5: Пульт оператора
            with gr.Tab("📋 Пульт оператора"):
                gr.Markdown("### Пульт оператора парковки")
                gr.Markdown("Обновляется автоматически. Данные — из базы PostgreSQL.")
                try:
                    _initial_source = _db().latest_source() or "main"
                except Exception:
                    _initial_source = "main"
                with gr.Row():
                    db_source = gr.Dropdown(label="Объект / камера",
                                            choices=[_initial_source], value=_initial_source)
                    db_refresh_btn = gr.Button("🔄 Обновить", variant="primary")

                db_summary = gr.Markdown("Загрузка...")
                db_board = gr.Dataframe(label="Состояние мест", interactive=False)
                db_journal = gr.Dataframe(label="Журнал посещений (последние 20)", interactive=False)

                db_outputs = [db_summary, db_board, db_journal, db_source]

                db_refresh_btn.click(fn=operator_refresh, inputs=db_source, outputs=db_outputs)

                with gr.Accordion("⚙️ Администрирование", open=False):
                    gr.Markdown("Первичная настройка: создать схему БД и загрузить зоны из `parking_zones.json`.")
                    db_init_btn = gr.Button("Инициализировать БД (схема + зоны)")
                    db_admin_status = gr.Textbox(label="Статус", interactive=False)
                    db_init_btn.click(fn=db_init, outputs=db_admin_status)

                db_timer = gr.Timer(5.0)
                db_timer.tick(fn=operator_refresh, inputs=db_source, outputs=db_outputs)

        gr.Markdown("---")
        gr.Markdown(
            "**Легенда:**\n"
            "- 🟢 Зелёная зона = свободно | 🔴 Красная = занято\n"
            "- 🔵 Синий bbox = транспорт | 🟣 Фиолетовый = другое\n"
            "- 🟠 Оранжевая штриховка = пересечение bbox с зоной"
        )

    return app


if __name__ == "__main__":
    print("Запуск Parking Monitor...")
    app = create_interface()
    app.launch()
