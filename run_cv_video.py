import cv2
import json
import os
import numpy as np
from ultralytics import YOLO

VIDEO_FILE = "vecteezy_parking-lot-movement_1623435.mp4"

model = YOLO('yolov8x.pt')
cap = cv2.VideoCapture(VIDEO_FILE)

ret, frame = cap.read()
if not ret:
    print("Ошибка")
    exit()

orig_h, orig_w = frame.shape[:2]
print(f"Оригинал: {orig_w}x{orig_h}")

cap.release()

# Масштаб для отображения
scale = 0.25
disp_w = int(orig_w * scale)
disp_h = int(orig_h * scale)

# Зоны
polygons = []
polygon_file = "parking_zones.json"

if os.path.exists(polygon_file):
    try:
        with open(polygon_file, 'r') as f:
            data = json.load(f)
            polygons = data.get('zones', [])
            print(f"Зон: {len(polygons)}")
    except:
        polygons = []

# Масштабируем зоны
scaled_zones = []
for z in polygons:
    if z.get('type') == 'rect':
        x1 = int(z['x1'] * scale)
        y1 = int(z['y1'] * scale)
        x2 = int(z['x2'] * scale)
        y2 = int(z['y2'] * scale)
        scaled_zones.append({'type': 'rect', 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})
    elif z.get('type') == 'poly':
        pts = [[int(p[0]*scale), int(p[1]*scale)] for p in z.get('points', [])]
        scaled_zones.append({'type': 'poly', 'points': pts})

print(f"Масштаб: {scale}, Зон: {len(scaled_zones)}")

cv2.namedWindow("RGB")
cap = cv2.VideoCapture(VIDEO_FILE)

points_stack = []

def RGB(event, x, y, flags, param):
    global points_stack, scaled_zones
    if event == cv2.EVENT_LBUTTONDOWN:
        points_stack.append((x, y))
        if len(points_stack) == 4:
            scaled_zones.append(points_stack.copy())
            points_stack.clear()
            print(f"Зона {len(scaled_zones)}")

cv2.setMouseCallback("RGB", RGB)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (disp_w, disp_h))
    results = model.track(frame, persist=True)

    for z in scaled_zones:
        if z.get('type') == 'rect':
            x1, y1, x2, y2 = z['x1'], z['y1'], z['x2'], z['y2']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        elif z.get('type') == 'poly':
            pts = np.array(z.get('points', []), np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    occupied = 0
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = box
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            for z in scaled_zones:
                if z.get('type') == 'rect':
                    x1, y1, x2, y2 = z['x1'], z['y1'], z['x2'], z['y2']
                    if x1 <= cx <= x2 and y1 <= cy <= y2:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        occupied += 1
                        break
                elif z.get('type') == 'poly':
                    pts = np.array(z.get('points', []), np.int32).reshape((-1, 1, 2))
                    if cv2.pointPolygonTest(pts, (cx, cy), False) >= 0:
                        cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
                        occupied += 1
                        break

    free = len(scaled_zones) - occupied
    cv2.putText(frame, f'FREE: {free}', (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'OCC: {occupied}', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    for pt in points_stack:
        cv2.circle(frame, pt, 5, (0, 255, 255), -1)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()