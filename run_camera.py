import cv2
import json
import os
import numpy as np
from ultralytics import YOLO

model = YOLO('yolov8x.pt')

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

polygon_points = []
polygons = []
polygon_file = "polygons.json"

if os.path.exists(polygon_file):
    try:
        with open(polygon_file, 'r') as f:
            polygons = json.load(f)
    except (json.JSONDecodeError, ValueError):
        polygons = []

def RGB(event, x, y, flags, param):
    global polygon_points, polygons
    if event == cv2.EVENT_LBUTTONDOWN:
        polygon_points.append((x, y))
        if len(polygon_points) == 4:
            polygons.append(polygon_points.copy())
            with open(polygon_file, 'w') as f:
                json.dump(polygons, f)
            polygon_points.clear()
            print(f"Зона сохранена. Всего зон: {len(polygons)}")

cv2.namedWindow("RGB")
cv2.setMouseCallback("RGB", RGB)

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 3 != 0:
        continue

    frame = cv2.resize(frame, (1020, 500))
    results = model.track(frame, persist=True)

    for poly in polygons:
        pts = np.array(poly, np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    occupied = 0
    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        boxes = results[0].boxes.xyxy.cpu().numpy()

        for track_id, box in zip(ids, boxes):
            x1, y1, x2, y2 = box
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            for poly in polygons:
                pts = np.array(poly, np.int32).reshape((-1, 1, 2))
                if cv2.pointPolygonTest(pts, (cx, cy), False) >= 0:
                    cv2.circle(frame, (cx, cy), 4, (255, 0, 255), -1)
                    cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
                    occupied += 1
                    break

    free = len(polygons) - occupied
    cv2.putText(frame, f'FREE: {free}', (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'OCC: {occupied}', (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    for pt in polygon_points:
        cv2.circle(frame, pt, 5, (0, 0, 255), -1)

    cv2.imshow("RGB", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        break
    elif key == ord('r') and polygons:
        polygons.pop()
        with open(polygon_file, 'w') as f:
            json.dump(polygons, f)

cap.release()
cv2.destroyAllWindows()