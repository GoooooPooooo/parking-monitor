"""Разметка парковочных зон на кадре видео (OpenCV).

Управление:
    ЛКМ           — 5 кликов на место: 4 угла + точка номера
    U             — отменить последнюю точку
    R             — удалить последнюю зону
    ESC           — выход (зоны сохраняются автоматически)

Зоны сохраняются в файл, отдельный для каждого видео: zones/<имя_видео>.json

Примеры:
    python setup_zones_cv.py --video image/Video/1790685044033-01a0ed25-3f0d-74be-b025-4a69044c279a.mp4
    python setup_zones_cv.py --video clip.mp4 --at 2.5 --scale 0.8
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np


def zones_path_for(video: str) -> str:
    """Путь к файлу зон для данного видео: zones/<stem>.json."""
    stem = os.path.splitext(os.path.basename(video))[0]
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "zones")
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, stem + ".json")


def load_zones(path: str) -> list:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f).get("zones", [])
        except (json.JSONDecodeError, ValueError):
            return []
    return []


def save_zones(path: str, zones: list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"zones": zones}, f, indent=2, ensure_ascii=False)


def zone_points(zone: dict):
    """Точки зоны (rect или poly) для отрисовки."""
    if zone.get("type") == "rect":
        x1, y1, x2, y2 = zone["x1"], zone["y1"], zone["x2"], zone["y2"]
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    return [tuple(p) for p in zone.get("points", [])]


def main():
    ap = argparse.ArgumentParser(description="Разметка зон на кадре видео")
    ap.add_argument("--video", required=True)
    ap.add_argument("--zones", default=None, help="по умолчанию zones/<имя_видео>.json")
    ap.add_argument("--at", type=float, default=0.0, help="секунда кадра для разметки")
    ap.add_argument("--scale", type=float, default=1.0, help="масштаб окна")
    args = ap.parse_args()

    zones_file = args.zones or zones_path_for(args.video)
    zones = load_zones(zones_file)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Не удалось открыть видео: {args.video}")
        sys.exit(1)
    if args.at > 0:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.at * fps))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Не удалось прочитать кадр")
        sys.exit(1)

    scale = args.scale
    current_points = []
    window = "Разметка зон"

    def to_frame_xy(x, y):
        return int(x / scale), int(y / scale)

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        fx, fy = to_frame_xy(x, y)
        current_points.append((fx, fy))
        print(f"Точка {len(current_points)}/5: ({fx}, {fy})")

        if len(current_points) == 5:
            corners = current_points[:4]
            label = current_points[4]
            xs = [p[0] for p in corners]
            ys = [p[1] for p in corners]
            zones.append({
                "type": "rect",
                "x1": min(xs), "y1": min(ys), "x2": max(xs), "y2": max(ys),
                "label_x": label[0], "label_y": label[1],
            })
            save_zones(zones_file, zones)
            print(f"Зона {len(zones)} сохранена -> {zones_file}")
            current_points.clear()

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)
    print(f"Файл зон: {zones_file}")
    print("5 кликов = зона (4 угла + номер) | U отмена точки | R удалить зону | ESC выход")

    while True:
        img = frame.copy()

        for i, z in enumerate(zones):
            pts = zone_points(z)
            if not pts:
                continue
            pts_np = np.array(pts, np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [pts_np], True, (0, 200, 0), 2)
            cx, cy = int(sum(p[0] for p in pts) / len(pts)), int(sum(p[1] for p in pts) / len(pts))
            cv2.putText(img, str(i + 1), (cx - 8, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)

        for j, pt in enumerate(current_points):
            cv2.circle(img, pt, 4, (0, 0, 255), -1)
            if j > 0:
                cv2.line(img, current_points[j - 1], pt, (0, 255, 255), 2)

        status = f"Zones: {len(zones)} | Points: {len(current_points)}/5"
        cv2.putText(img, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        shown = img if scale == 1.0 else cv2.resize(img, None, fx=scale, fy=scale)
        cv2.imshow(window, shown)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key in (ord("r"), ord("R")) and zones:
            zones.pop()
            save_zones(zones_file, zones)
            print(f"Зона удалена. Всего: {len(zones)}")
        elif key in (ord("u"), ord("U"), 8) and current_points:
            current_points.pop()
            print(f"Точка отменена. Осталось: {len(current_points)}")

    cv2.destroyAllWindows()
    save_zones(zones_file, zones)
    print(f"Готово. Зон: {len(zones)} -> {zones_file}")


if __name__ == "__main__":
    main()
