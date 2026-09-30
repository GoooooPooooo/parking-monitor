"""Видеомониторинг: детекция транспорта и подсчёт занятости по зонам.

Зоны берутся из файла зон для этого видео: zones/<имя_видео>.json (или --zones).
Управление: ESC — выход.

Примеры:
    python run_cv_video.py --video image/Video/1790685044033-01a0ed25-3f0d-74be-b025-4a69044c279a.mp4
    python run_cv_video.py --video clip.mp4 --scale 0.8 --loop
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

import config
from detector import CarDetector
from zones import ParkingZones


def zones_path_for(video: str) -> str:
    stem = os.path.splitext(os.path.basename(video))[0]
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "zones")
    return os.path.join(folder, stem + ".json")


def main():
    ap = argparse.ArgumentParser(description="Видеомониторинг парковки")
    ap.add_argument("--video", default=config.VIDEO_FILE)
    ap.add_argument("--zones", default=None, help="по умолчанию zones/<имя_видео>.json")
    ap.add_argument("--scale", type=float, default=1.0, help="масштаб окна")
    ap.add_argument("--model", default=None, help="переопределить модель из config.py")
    ap.add_argument("--loop", action="store_true", help="проигрывать видео по кругу")
    ap.add_argument("--db", action="store_true", help="писать снимки занятости в PostgreSQL")
    ap.add_argument("--source", default=None, help="имя источника в БД (по умолчанию имя видео)")
    ap.add_argument("--db-interval", type=float, default=2.0, help="интервал записи в БД, сек")
    args = ap.parse_args()

    zones_file = args.zones or zones_path_for(args.video)
    if not os.path.exists(zones_file):
        fallback = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parking_zones.json")
        if os.path.exists(fallback):
            print(f"Зоны для этого видео не найдены, беру {fallback}")
            zones_file = fallback
    if not os.path.exists(zones_file):
        print(f"Файл зон не найден: {zones_file}")
        print("Сначала разметьте зоны:")
        print(f"  python setup_zones_cv.py --video {args.video}")
        sys.exit(1)

    if args.model:
        config.YOLO_MODEL = args.model
    config.DEBUG_MODE = False  # не засорять консоль на каждом кадре

    detector = CarDetector()
    parking = ParkingZones(zones_file)
    print(f"Модель: {config.YOLO_MODEL} | Зон: {len(parking.zones)}")

    # Запись в БД
    use_db = args.db
    db_ids: list[int] = []
    db_source = args.source or os.path.splitext(os.path.basename(args.video))[0]
    repository = None
    if use_db:
        try:
            from db import repository
            repository.init_schema()
            db_ids = repository.sync_zones(parking.zones, source=db_source)
            print(f"БД: источник='{db_source}', зон записано={len(db_ids)}")
        except Exception as e:
            print(f"⚠️ БД недоступна ({e}) — продолжаю без записи")
            use_db = False

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Не удалось открыть видео: {args.video}")
        sys.exit(1)

    window = "Parking Monitor"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    last_db = 0.0
    while True:
        ret, frame = cap.read()
        if not ret:
            if args.loop:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = detector.detect_with_tracking(rgb, persist=True)
        parking.update_states_by_center(detections)

        if use_db and time.time() - last_db >= args.db_interval:
            states = [
                {"zone_id": db_ids[i], "is_occupied": bool(parking.zone_states[i])}
                for i in range(min(len(db_ids), len(parking.zone_states)))
            ]
            try:
                snap_id = repository.log_snapshot(states, source=db_source, model=config.YOLO_MODEL)
                free, occ, total = parking.get_counts()
                print(f"[{time.strftime('%H:%M:%S')}] БД #{snap_id}: "
                      f"свободно {free}/{total}, занято {occ}")
            except Exception as e:
                print(f"⚠️ Ошибка записи в БД: {e}")
            last_db = time.time()

        for i, zone in enumerate(parking.zones):
            pts = parking._get_zone_points(zone)
            if not pts:
                continue
            pts_np = np.array(pts, np.int32).reshape((-1, 1, 2))
            color = (0, 0, 255) if parking.zone_states[i] else (0, 200, 0)
            cv2.polylines(frame, [pts_np], True, color, 2)
            cx, cy = int(pts_np[:, 0, 0].mean()), int(pts_np[:, 0, 1].mean())
            cv2.putText(frame, str(i + 1), (cx - 6, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        for det in detections:
            x1, y1, x2, y2 = (int(v) for v in det[:4])
            conf = det[4]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
            cv2.putText(frame, f"{conf:.2f}", (x1, max(y1 - 4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        free, occupied, total = parking.get_counts()
        cv2.rectangle(frame, (0, 0), (330, 40), (0, 0, 0), -1)
        cv2.putText(frame, f"Free: {free}  Occupied: {occupied}  Total: {total}",
                    (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        shown = frame if args.scale == 1.0 else cv2.resize(frame, None, fx=args.scale, fy=args.scale)
        cv2.imshow(window, shown)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
