import cv2
import sys

video = "vecteezy_parking-lot-movement_1623435.mp4"
scale = 0.25  # Масштаб

cap = cv2.VideoCapture(video)
if not cap.isOpened():
    print("Ошибка")
    sys.exit()

ret, frame = cap.read()
if not ret:
    print("Ошибка чтения")
    exit()

orig_h, orig_w = frame.shape[:2]
cap.release()

disp_w = int(orig_w * scale)
disp_h = int(orig_h * scale)

cap = cv2.VideoCapture(video)
ret, frame = cap.read()
cap.release()

frame_small = cv2.resize(frame, (disp_w, disp_h))
cv2.imwrite("parking_frame.jpg", frame_small)
print(f"Сохранено: parking_frame.jpg ({disp_w}x{disp_h})")