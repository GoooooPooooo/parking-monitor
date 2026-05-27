import cv2
import json
import os
import numpy as np

polygon_file = "parking_zones.json"
zones = []

if os.path.exists(polygon_file):
    try:
        with open(polygon_file, 'r') as f:
            zones = json.load(f).get('zones', [])
    except:
        zones = []

current_points = []
window_name = "Разметка зон"

cap = cv2.VideoCapture("vecteezy_parking-lot-movement_1623435.mp4")
ret, frame = cap.read()
cap.release()

if not ret:
    print("Ошибка чтения видео")
    exit()

frame = cv2.resize(frame, (1020, 500))

def save():
    with open(polygon_file, 'w') as f:
        json.dump({'zones': zones}, f, indent=2)

def on_mouse(event, x, y, flags, param):
    global current_points, zones
    if event == cv2.EVENT_LBUTTONDOWN:
        current_points.append((x, y))
        print(f"Точка {len(current_points)}/5: ({x}, {y})")
        
        if len(current_points) == 5:
            corners = current_points[:4]
            label = current_points[4]
            xs = [p[0] for p in corners]
            ys = [p[1] for p in corners]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            
            zone = {
                "type": "rect",
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "label_x": label[0], "label_y": label[1]
            }
            zones.append(zone)
            save()
            print(f"✅ Зона {len(zones)} сохранена")
            current_points = []

cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, on_mouse)

print("Разметка: 4 угла + 1 номер = зона | R = удалить | ESC = выход")

while True:
    img = frame.copy()
    
    for i, z in enumerate(zones):
        if z.get('type') == 'rect':
            x1, y1, x2, y2 = z['x1'], z['y1'], z['x2'], z['y2']
            cx, cy = (x1+x2)//2, (y1+y2)//2
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(img, (cx, cy), 10, (255, 0, 0), -1)
            cv2.putText(img, str(i+1), (cx-5, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    for pt in current_points:
        cv2.circle(img, pt, 5, (0, 0, 255), -1)
        if len(current_points) > 1:
            for j in range(len(current_points)-1):
                cv2.line(img, current_points[j], current_points[j+1], (0, 255, 255), 2)
    
    status = f"Зон: {len(zones)} | Точек: {len(current_points)}/5"
    cv2.putText(img, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(img, "4 угла + номер | R = удалить | ESC = выход", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    cv2.imshow(window_name, img)
    
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == ord('r') and zones:
        zones.pop()
        save()
        print(f"Удалено. Зон: {len(zones)}")

cv2.destroyAllWindows()
print(f"Готово! Зон: {len(zones)}")