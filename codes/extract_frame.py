"""Извлечь кадр из видео (для разметки зон).

Примеры:
    python extract_frame.py
    python extract_frame.py --video image/Video/1790685044033-01a0ed25-3f0d-74be-b025-4a69044c279a.mp4 --out parking_frame.jpg --at 1.0
"""

import argparse

import cv2


def main():
    ap = argparse.ArgumentParser(description="Извлечь кадр из видео")
    ap.add_argument("--video", default="vecteezy_parking-lot-movement_1623435.mp4")
    ap.add_argument("--out", default="parking_frame.jpg")
    ap.add_argument("--at", type=float, default=0.0, help="секунда, с которой взять кадр")
    ap.add_argument("--scale", type=float, default=1.0, help="масштаб сохранения")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Не удалось открыть видео: {args.video}")
        return

    if args.at > 0:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.at * fps))

    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Не удалось прочитать кадр")
        return

    if args.scale != 1.0:
        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (int(w * args.scale), int(h * args.scale)))

    cv2.imwrite(args.out, frame)
    print(f"Сохранено: {args.out} ({frame.shape[1]}x{frame.shape[0]})")


if __name__ == "__main__":
    main()
