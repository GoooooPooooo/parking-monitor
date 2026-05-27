import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(10, 8))
ax.set_xlim(0, 1400)
ax.set_ylim(0, 1100)
ax.axis("off")

BOX_W = 200
BOX_H = 60
ARROW_Y = 30
COLORS = {
    "ui": "#4A90D9",
    "module": "#50B86C",
    "data": "#D9A05B",
    "model": "#D9645B",
    "storage": "#7B68AE",
    "text": "#2C3E50",
}


def center(x, y, w, h):
    return x - w / 2, y - h / 2, w, h


def box(cx, cy, w, h, color, text, text_color="white", fontsize=10):
    x, y, _, _ = center(cx, cy, w, h)
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.1",
        facecolor=color, edgecolor="#333", linewidth=1.5, zorder=2
    ))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color=text_color, zorder=3)


def arrow(x1, y1, x2, y2, label="", style="->"):
    dx = x2 - x1
    dy = y2 - y1
    ax.annotate("", xy=(x2, y2 - ARROW_Y if dy < 0 else y2),
                xytext=(x1, y1 + ARROW_Y if dy > 0 else y1),
                arrowprops=dict(arrowstyle=style, color="#555", lw=1.5))
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my - 22 if dy < 0 else my + 22, label,
                ha="center", va="center", fontsize=8, color="#555",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=2))


# ── Layer: UI ──
box(700, 980, 360, 70, COLORS["ui"], "Gradio Web UI\n(main.py — 4 вкладки)")
box(360, 980, 200, 70, COLORS["ui"], "OpenCV Окно\n(run_cv_video.py)", fontsize=9)
box(1040, 980, 200, 70, COLORS["ui"], "Tkinter Разметка\n(setup_zones.py)", fontsize=9)

# ── Label: UI Layer ──
ax.text(50, 1020, "УРОВЕНЬ\nПОЛЬЗОВАТЕЛЯ", fontsize=9,
        fontweight="bold", color="#888", va="top")

# ── Layer: Modules ──
bx_center = 700
by_modules = 780

box(bx_center - 280, by_modules, BOX_W, BOX_H, COLORS["module"], "detector.py\nYOLO-обёртка")
box(bx_center, by_modules, BOX_W, BOX_H, COLORS["module"], "zones.py\nПарковочные зоны")
box(bx_center + 280, by_modules, BOX_W, BOX_H, COLORS["module"], "config.py\nПараметры + пороги")

ax.text(50, 820, "МОДУЛИ\nЯДРА", fontsize=9,
        fontweight="bold", color="#888", va="top")

# ── Layer: Data / Models ──
by_data = 580

box(bx_center - 280, by_data, BOX_W, BOX_H, COLORS["data"], "parking_zones.json\nКоординаты зон", fontsize=9)
box(bx_center, by_data, BOX_W, BOX_H, COLORS["model"], "YOLOv8 Model\n*.pt / ONNX", fontsize=9)
box(bx_center + 280, by_data, BOX_W, BOX_H, COLORS["data"], "polygons.json\nПолигоны\n(OpenCV)", fontsize=9)

ax.text(50, 620, "ДАННЫЕ И\nМОДЕЛЬ", fontsize=9,
        fontweight="bold", color="#888", va="top")

# ── Layer: MLflow ──
box(700, 380, 300, 70, COLORS["storage"], "MLflow\nЭксперименты, метрики, артефакты", fontsize=9)
by_storage = 380
ax.text(50, 420, "МОНИТОРИНГ\nОБУЧЕНИЯ", fontsize=9,
        fontweight="bold", color="#888", va="top")

# ── Arrows: UI → Modules ──
arrow(700, 945, bx_center - 280, by_modules + BOX_H / 2 + ARROW_Y)
arrow(700, 945, bx_center, by_modules + BOX_H / 2 + ARROW_Y)
arrow(700, 945, bx_center + 280, by_modules + BOX_H / 2 + ARROW_Y)

# ── Arrows: Modules → Data ──
arrow(bx_center - 280, by_modules - BOX_H / 2, bx_center - 280, by_data + BOX_H / 2 + ARROW_Y)
arrow(bx_center, by_modules - BOX_H / 2, bx_center, by_data + BOX_H / 2 + ARROW_Y, "загружает")
arrow(bx_center + 280, by_modules - BOX_H / 2, bx_center + 280, by_data + BOX_H / 2 + ARROW_Y)

# ── Arrows: Data → MLflow ──
arrow(bx_center, by_data - BOX_H / 2, 700, by_storage + BOX_H / 2 + ARROW_Y, "логирование метрик")

# ── Finetune module ──
box(700, 180, 300, 70, COLORS["storage"], "finetune.py\nKaggle датасет → Дообучение", fontsize=9)
ax.text(50, 220, "ДООБУЧЕНИЕ", fontsize=9,
        fontweight="bold", color="#888", va="top")
arrow(700, 215, 700, by_storage + BOX_H / 2 + ARROW_Y, "сохраняет веса")

# ── Title ──
ax.text(700, 1080, "Архитектура системы мониторинга парковки",
        ha="center", va="center", fontsize=16, fontweight="bold", color="#2C3E50")

plt.savefig("docs/architecture.png", dpi=150, bbox_inches="tight",
            facecolor="white", edgecolor="none")
print("OK - docs/architecture.png generated")
