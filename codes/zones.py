"""Управление зонами парковочных мест (метод центра объекта)."""

import json
import os
import numpy as np
import cv2
from PIL import ImageDraw, Image
import config


class ParkingZones:
    """Управление парковочными зонами - проверка центра объекта в полигоне."""

    def __init__(self, zones_file=None):
        self.zones = []  # список dict с точками полигонов
        self.zone_states = []  # True = занято, False = свободно

        if zones_file and os.path.exists(zones_file):
            self.load(zones_file)
        else:
            self._create_default_zones()

    def _create_default_zones(self):
        self.zones = []
        self.zone_states = []

    def load(self, filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.zones = data.get('zones', [])
        self.zone_states = [False] * len(self.zones)

    def save(self, filepath):
        data = {'zones': self.zones}
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def update_states_by_center(self, detections):
        """Обновить состояния зон - центр + bbox (комбинированный метод)."""
        self.zone_states = [False] * len(self.zones)

        for det in detections:
            x1, y1, x2, y2 = det[:4]
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            for zone_idx, zone in enumerate(self.zones):
                pts = self._get_zone_points(zone)
                if pts is None:
                    continue
                pts_np = np.array(pts, np.int32).reshape((-1, 1, 2))

                # 1. Проверяем центр
                if cv2.pointPolygonTest(pts_np, (cx, cy), False) >= 0:
                    self.zone_states[zone_idx] = True
                    continue

                # 2. Проверяем пересечение bbox с зоной
                if self._bbox_intersects_zone(x1, y1, x2, y2, pts_np):
                    self.zone_states[zone_idx] = True
                    continue

    def _bbox_intersects_zone(self, x1, y1, x2, y2, zone_pts):
        """Проверить пересечение bbox с зоной."""
        # Проверяем 4 угла bbox
        corners = [(x1, y1), (x1, y2), (x2, y1), (x2, y2)]
        for px, py in corners:
            if cv2.pointPolygonTest(zone_pts, (px, py), False) >= 0:
                return True

        # Проверяем пересечение линий (упрощенно: проверяем находится ли любой угол bbox в зоне ИЛИ зона пересекает bbox)
        # Проверяем середины сторон bbox
        mid_points = [
            ((x1 + x2) / 2, y1),  # top mid
            ((x1 + x2) / 2, y2),  # bottom mid
            (x1, (y1 + y2) / 2),  # left mid
            (x2, (y1 + y2) / 2),  # right mid
        ]
        for px, py in mid_points:
            if cv2.pointPolygonTest(zone_pts, (px, py), False) >= 0:
                return True

        return False

    def _get_zone_points(self, zone):
        """Получить точки зоны для OpenCV."""
        if isinstance(zone, dict):
            if zone.get("type") == "rect":
                x1, y1, x2, y2 = zone["x1"], zone["y1"], zone["x2"], zone["y2"]
                return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            else:
                return zone.get("points", [])
        return []

    def get_counts(self):
        total = len(self.zones)
        occupied = sum(self.zone_states)
        return total - occupied, occupied, total

    def draw_zones(self, image, detections=None):
        """Отрисовать зоны (красный=занято, зеленый=свободно)."""
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        draw = ImageDraw.Draw(image)

        for i, zone in enumerate(self.zones):
            color = "red" if self.zone_states[i] else "green"
            width = 3 if self.zone_states[i] else 2

            pts = self._get_zone_points(zone)
            if pts:
                pts_tuple = [tuple(p) for p in pts]
                draw.polygon(pts_tuple, outline=color, width=width)
                cx = sum(p[0] for p in pts) // 4
                cy = sum(p[1] for p in pts) // 4
                label = str(i + 1)
                draw.text((cx - 5, cy - 7), label, fill=color)

        return image


def zone_polygon_points(zone):
    """Преобразовать зону в список точек [[x, y], ...] для OpenCV."""
    if isinstance(zone, dict):
        if zone.get("type") == "rect":
            x1, y1 = zone["x1"], zone["y1"]
            x2, y2 = zone["x2"], zone["y2"]
            return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        return zone.get("points", [])
    return zone


def load_zones_file(filepath):
    """Загрузить зоны из JSON: формат {'zones': [...]} или плоский список."""
    if not filepath or not os.path.exists(filepath):
        return []
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except (json.JSONDecodeError, ValueError):
        return []
    if isinstance(data, dict):
        return data.get('zones', [])
    if isinstance(data, list):
        return [{'type': 'poly', 'points': [list(p) for p in z]} for z in data]
    return []


def save_zones_file(filepath, zones):
    """Сохранить зоны в JSON в едином формате."""
    with open(filepath, 'w') as f:
        json.dump({'zones': zones}, f, indent=2)
