"""
Инструмент для интерактивной разметки парковочных зон.
Запуск: python setup_zones.py

Поток:
  1. Камера показывает видеопоток
  2. Нажмите SPACE — сделать снимок
  3. Кликайте 4 угла парковочного места
  4. 5-й клик — где номер места (важно!)
  5. Повторите для всех 8 мест
  6. Нажмите 'S' для сохранения в parking_zones.json
  7. Нажмите 'Q' для выхода
"""

import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import json
import config
import sys


class ZoneEditor:
    """Интерактивный редактор парковочных зон с камерой."""

    def __init__(self):
        self.zones = []
        self.current_points = []  # 4 угла + 1 номер
        self.frozen = False
        self.frozen_frame = None
        self.tk_image = None
        self.scale = 1.0
        self.current_frame = None
        self.orig_h = 0
        self.orig_w = 0

        # Видео
        print("Открытие видео...")
        self.video_file = "vecteezy_parking-lot-movement_1623435.mp4"
        self.cap = cv2.VideoCapture(self.video_file)

        if not self.cap.isOpened():
            print("ОШИБКА: Не удалось открыть видео!")
            messagebox.showerror("Ошибка", "Не удалось открыть видео!")
            sys.exit(1)

        ret, test_frame = self.cap.read()
        if not ret:
            print("ОШИБКА: Видео открыто, но не читает кадры!")
            sys.exit(1)

        print(f"Видео готово: {test_frame.shape[1]}x{test_frame.shape[0]}")

        # GUI
        self.root = tk.Tk()
        self.root.title("Разметка парковочных зон — Камера")
        self.root.protocol("WM_DELETE_WINDOW", self.quit_app)

        # Canvas
        canvas_width, canvas_height = 800, 600
        self.canvas = tk.Canvas(self.root, width=canvas_width, height=canvas_height, bg="black")
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_click)

        # Инструкция
        self.info_label = tk.Label(
            self.root,
            text="SPACE = снимок → 4 угла места → номер места | S = сохранить | Q = выход",
            font=("Arial", 11)
        )
        self.info_label.pack(pady=5)

        # Привязка клавиш
        self.root.bind("<space>", self.take_snapshot)
        self.root.bind("s", self.save)
        self.root.bind("S", self.save)
        self.root.bind("q", self.quit_app)
        self.root.bind("Q", self.quit_app)
        self.root.bind("z", self.undo)
        self.root.bind("Z", self.undo)

        print("Запуск окна...")
        self.update_frame()
        self.root.mainloop()

    def update_frame(self):
        """Обновление видеопотока."""
        if not self.frozen:
            ret, frame = self.cap.read()
            if ret:
                self.show_frame(frame)
            else:
                print("Не удалось прочитать кадр")

        self.root.after(30, self.update_frame)

    def show_frame(self, frame):
        """Отобразить кадр на canvas."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.current_frame = frame_rgb
        self.orig_h, self.orig_w = frame.shape[:2]

        canvas_width = 800
        canvas_height = 600

        self.scale = min(canvas_width / self.orig_w, canvas_height / self.orig_h)
        new_w = int(self.orig_w * self.scale)
        new_h = int(self.orig_h * self.scale)

        img = Image.fromarray(frame_rgb).resize((new_w, new_h), Image.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(img)

        self.canvas.delete("video")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image, tag="video")

    def take_snapshot(self, event=None):
        """Зафиксировать кадр для разметки."""
        if self.current_frame is None:
            print("Нет кадра для фиксации")
            return

        self.frozen = True

        canvas_width = 800
        canvas_height = 600

        new_w = int(self.orig_w * self.scale)
        new_h = int(self.orig_h * self.scale)

        img = Image.fromarray(self.current_frame).resize((new_w, new_h), Image.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(img)

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

        print("📸 Снимок зафиксирован!")
        self.update_info()

    def on_click(self, event):
        """Обработка клика."""
        if not self.frozen:
            print("Сначала нажмите SPACE для снимка")
            return

        # Координаты относительно оригинального кадра
        x = int(event.x / self.scale)
        y = int(event.y / self.scale)

        num_point = len(self.current_points) + 1

        if num_point <= 4:
            print(f"Угол {num_point}/4: ({x}, {y})")
            # Рисуем точку угла
            r = 4
            self.canvas.create_oval(
                event.x - r, event.y - r, event.x + r, event.y + r,
                fill="yellow", outline="black"
            )
        elif num_point == 5:
            print(f"Номер места: ({x}, {y})")
            # Рисуем номер большим кружком
            r = 8
            self.canvas.create_oval(
                event.x - r, event.y - r, event.x + r, event.y + r,
                fill="cyan", outline="black"
            )

        self.current_points.append((x, y))

        # Если набрано 5 точек (4 угла + номер) — создаём зону
        if len(self.current_points) == 5:
            self._create_zone()

        self.update_info()

    def _create_zone(self):
        """Создать зону из 5 точек (4 угла + номер)."""
        corners = self.current_points[:4]
        label_x, label_y = self.current_points[4]

        xs = [p[0] for p in corners]
        ys = [p[1] for p in corners]

        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        zone = (x1, y1, x2, y2, label_x, label_y)
        self.zones.append(zone)

        print(f"✅ Зона {len(self.zones)}: bbox=({x1},{y1},{x2},{y2}) label=({label_x},{label_y})")

        # Рисуем прямоугольник
        self._draw_zone_rect(zone, len(self.zones))

        self.current_points = []

        if len(self.zones) >= config.NUM_PARKING_SPOTS:
            print(f"🎉 Все {config.NUM_PARKING_SPOTS} зон созданы! Нажмите S для сохранения.")

    def _draw_zone_rect(self, zone, zone_num):
        """Нарисовать прямоугольник зоны."""
        x1, y1, x2, y2, label_x, label_y = zone
        sx1, sy1 = int(x1 * self.scale), int(y1 * self.scale)
        sy2 = int(y2 * self.scale)
        sx2 = int(x2 * self.scale)
        slx = int(label_x * self.scale)
        sly = int(label_y * self.scale)

        self.canvas.create_rectangle(sx1, sy1, sx2, sy2, outline="green", width=2, tag="zones")

        # Номер кружком
        r = 12
        self.canvas.create_oval(slx - r, sly - r, slx + r, sly + r,
                                fill="cyan", outline="white", width=2, tag="zones")
        self.canvas.create_text(slx, sly + 2, text=str(zone_num), fill="black",
                                font=("Arial", 12, "bold"), tag="zones")

    def undo(self, event=None):
        """Отменить последнюю точку."""
        if self.current_points:
            self.current_points.pop()
            print(f"Отмена. Осталось точек: {len(self.current_points)}")
            self.update_info()

    def save(self, event=None):
        """Сохранить зоны в JSON."""
        if not self.zones:
            print("Нет зон для сохранения!")
            return

        filepath = "parking_zones.json"
        zones_list = []
        for z in self.zones:
            x1, y1, x2, y2, lx, ly = z
            zones_list.append({
                "type": "rect",
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "label_x": lx, "label_y": ly
            })
        with open(filepath, 'w') as f:
            json.dump({'zones': zones_list}, f, indent=2)

        print(f"✅ Сохранено {len(self.zones)} зон в {filepath}")

    def update_info(self):
        """Обновить информацию."""
        num_in_current = len(self.current_points)
        status = "РЕЖИМ ВИДЕО" if not self.frozen else f"РАЗМЕТКА ({len(self.zones)}/{config.NUM_PARKING_SPOTS})"

        if not self.frozen:
            text = f"{status} | SPACE=снимок | S=сохранить | Q=выход"
        elif num_in_current == 0:
            text = f"{status} | Кликайте 4 угла места | Z=отменить"
        elif num_in_current < 4:
            text = f"{status} | Углы: {num_in_current}/4 | Z=отменить"
        elif num_in_current == 4:
            text = f"{status} | Углы готовы → кликните где НОМЕР места | Z=отменить"
        else:
            text = f"{status} | Зон: {len(self.zones)} | SPACE=переснять | S=сохранить | Q=выход"

        self.info_label.config(text=text)

    def quit_app(self, event=None):
        """Выход."""
        print("Выход...")
        self.cap.release()
        self.root.quit()
        self.root.destroy()


if __name__ == "__main__":
    print("Запуск редактора зон...")
    editor = ZoneEditor()
