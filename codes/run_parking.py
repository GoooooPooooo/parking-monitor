import cv2
import numpy as np
from ultralytics import YOLO
from zones import load_zones_file, zone_polygon_points, save_zones_file
import config

model = YOLO('models/yolov8x.pt')

VIDEO_FILE = "output_parking.mp4"
cap = cv2.VideoCapture(VIDEO_FILE)

polygon_points = []
polygon_file = "parking_zones.json"  # Как на YouTube
zones = load_zones_file(polygon_file)
polygons = [zone_polygon_points(z) for z in zones]
print(f"Загружено зон: {len(polygons)}")

def RGB(event, x, y, flags, param):
    global polygon_points, polygons, zones
    if event == cv2.EVENT_LBUTTONDOWN:
        polygon_points.append((x, y))
        if len(polygon_points) == 4:
            pts = polygon_points.copy()
            polygons.append(pts)
            zones.append({
                'type': 'poly',
                'points': [list(p) for p in pts],
                'label_x': sum(p[0] for p in pts) // 4,
                'label_y': sum(p[1] for p in pts) // 4,
            })
            save_zones_file(polygon_file, zones)
            polygon_points.clear()

cv2.namedWindow("RGB")
cv2.setMouseCallback("RGB", RGB)

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        if not ret:
            break

    frame = cv2.resize(frame, (1020, 500))
    results = model.track(frame, persist=True)

    for poly in polygons:
        pts = np.array(poly, np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    occupied = 0
    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy().astype(int)

        for track_id, box, cls in zip(ids, boxes, classes):
            if config.DETECT_CLASSES is not None and cls not in config.DETECT_CLASSES:
                continue
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

    cv2.imshow("RGB", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == ord('r') and polygons:
        polygons.pop()
        if zones:
            zones.pop()
        save_zones_file(polygon_file, zones)

cap.release()
cv2.destroyAllWindows()