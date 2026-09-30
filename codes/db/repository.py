"""Операции с БД: схема, зоны, снимки, сессии стоянки, табло оператора."""

import json
import os
from datetime import datetime, timezone
from typing import Iterable

import pandas as pd
from sqlalchemy import text

from db.connection import get_engine

SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema.sql")
ZONES_JSON = os.path.join(os.path.dirname(__file__), "..", "parking_zones.json")


# ==================== СХЕМА ====================

def init_schema() -> bool:
    """Применить DDL из schema.sql (идемпотентно)."""
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        ddl = f.read()
    with get_engine().begin() as conn:
        conn.execute(text(ddl))
    return True


# ==================== ЗОНЫ ====================

def sync_zones(zones: list, source: str = "main") -> list[int]:
    """Синхронизировать список зон и вернуть их id в исходном порядке."""
    sql = text(
        """
        INSERT INTO parking_zones (source, name, zone_type, points, label_x, label_y)
        VALUES (:source, :name, :zone_type, CAST(:points AS JSONB), :label_x, :label_y)
        ON CONFLICT (source, name) DO UPDATE
            SET zone_type = EXCLUDED.zone_type,
                points    = EXCLUDED.points,
                label_x   = EXCLUDED.label_x,
                label_y   = EXCLUDED.label_y
        RETURNING id
        """
    )
    ids: list[int] = []
    with get_engine().begin() as conn:
        for i, z in enumerate(zones, start=1):
            ids.append(
                conn.execute(
                    sql,
                    {
                        "source": source,
                        "name": f"Spot {i}",
                        "zone_type": z.get("type", "poly"),
                        "points": json.dumps(z.get("points")),
                        "label_x": z.get("label_x"),
                        "label_y": z.get("label_y"),
                    },
                ).scalar_one()
            )
    return ids


def sync_zones_from_json(path: str = ZONES_JSON, source: str = "main") -> int:
    """Перенести зоны из parking_zones.json в таблицу parking_zones."""
    with open(path, "r", encoding="utf-8") as f:
        zones = json.load(f).get("zones", [])
    sync_zones(zones, source=source)
    return len(zones)


def get_zones(source: str = "main") -> list[dict]:
    return _query(
        "SELECT id, name, zone_type, points, label_x, label_y "
        "FROM parking_zones WHERE source = :src AND is_active ORDER BY id",
        {"src": source},
    )


def list_sources() -> list[str]:
    return [r["source"] for r in _query("SELECT DISTINCT source FROM parking_zones ORDER BY source")]


def latest_source() -> str | None:
    """Источник с самым свежим снимком (чтобы табло показывало актуальные данные)."""
    rows = _query("SELECT source FROM occupancy_snapshots ORDER BY captured_at DESC LIMIT 1")
    return rows[0]["source"] if rows else None


# ==================== СНИМКИ И СЕССИИ ====================

def log_snapshot(states: Iterable[dict], source: str = "main", model: str | None = None) -> int:
    """Записать снимок занятости и обновить сессии (приезд/уезд).

    states: список словарей {'zone_id': int, 'is_occupied': bool, 'confidence': float|None}.
    Возвращает id созданного снимка.
    """
    states = list(states)
    total = len(states)
    occupied = sum(1 for s in states if s.get("is_occupied"))
    free = total - occupied

    with get_engine().begin() as conn:
        snapshot_id = conn.execute(
            text(
                """
                INSERT INTO occupancy_snapshots (source, model, total, occupied, free)
                VALUES (:source, :model, :total, :occupied, :free)
                RETURNING id
                """
            ),
            {"source": source, "model": model, "total": total, "occupied": occupied, "free": free},
        ).scalar_one()

        row_sql = text(
            """
            INSERT INTO zone_occupancy (snapshot_id, zone_id, is_occupied, confidence)
            VALUES (:snapshot_id, :zone_id, :is_occupied, :confidence)
            """
        )
        for s in states:
            conn.execute(
                row_sql,
                {
                    "snapshot_id": snapshot_id,
                    "zone_id": s["zone_id"],
                    "is_occupied": bool(s["is_occupied"]),
                    "confidence": s.get("confidence"),
                },
            )
            _apply_session(conn, s["zone_id"], bool(s["is_occupied"]))
    return snapshot_id


def _apply_session(conn, zone_id: int, is_occupied: bool) -> None:
    """Открыть сессию при появлении машины, закрыть — при отъезде."""
    if is_occupied:
        exists = conn.execute(
            text("SELECT 1 FROM parking_sessions WHERE zone_id = :z AND is_open LIMIT 1"),
            {"z": zone_id},
        ).first()
        if not exists:
            conn.execute(
                text("INSERT INTO parking_sessions (zone_id) VALUES (:z)"),
                {"z": zone_id},
            )
    else:
        conn.execute(
            text(
                "UPDATE parking_sessions SET departed_at = now(), is_open = FALSE "
                "WHERE zone_id = :z AND is_open"
            ),
            {"z": zone_id},
        )


# ==================== ТАБЛО ОПЕРАТОРА ====================

_RAW_BOARD_SQL = """
SELECT z.name AS spot,
       (open_s.id IS NOT NULL) AS is_occupied,
       open_s.arrived_at       AS arrived_at,
       last_s.arrived_at       AS last_arrived_at,
       last_s.departed_at      AS last_departed_at
FROM parking_zones z
LEFT JOIN LATERAL (
    SELECT s.id, s.arrived_at FROM parking_sessions s
    WHERE s.zone_id = z.id AND s.is_open
    ORDER BY s.arrived_at DESC LIMIT 1
) open_s ON TRUE
LEFT JOIN LATERAL (
    SELECT s.arrived_at, s.departed_at FROM parking_sessions s
    WHERE s.zone_id = z.id
    ORDER BY s.arrived_at DESC LIMIT 1
) last_s ON TRUE
WHERE z.source = :src AND z.is_active
ORDER BY z.id
"""


def operator_board(source: str = "main") -> list[dict]:
    """Текущее состояние мест (для сайта — на русском)."""
    rows = _query(_RAW_BOARD_SQL, {"src": source})
    now = datetime.now(timezone.utc)
    board = []
    for r in rows:
        spot = _ru_spot(r["spot"])
        if r["is_occupied"]:
            board.append({
                "Место": spot,
                "Статус": "Занято",
                "Приезд": _fmt_dt(r["arrived_at"]),
                "Уезд": "—",
                "Стоит": _fmt_dur(now - r["arrived_at"] if r["arrived_at"] else None),
            })
        else:
            board.append({
                "Место": spot,
                "Статус": "Свободно",
                "Приезд": _fmt_dt(r["last_arrived_at"]),
                "Уезд": _fmt_dt(r["last_departed_at"]),
                "Стоит": "—",
            })
    return board


def sessions_journal(limit: int = 20, source: str = "main") -> list[dict]:
    """Журнал посещений (сессии стоянки)."""
    rows = _query(
        "SELECT z.name AS spot, s.arrived_at, s.departed_at, s.is_open, "
        "COALESCE(s.departed_at, now()) - s.arrived_at AS duration "
        "FROM parking_sessions s JOIN parking_zones z ON z.id = s.zone_id "
        "WHERE z.source = :src ORDER BY s.arrived_at DESC LIMIT :lim",
        {"src": source, "lim": limit},
    )
    return [
        {
            "Место": _ru_spot(r["spot"]),
            "Приезд": _fmt_dt(r["arrived_at"]),
            "Уезд": _fmt_dt(r["departed_at"]),
            "Длительность": _fmt_dur(r["duration"]),
            "Статус": "На месте" if r["is_open"] else "Уехал",
        }
        for r in rows
    ]


def board_summary(source: str = "main") -> dict:
    board = operator_board(source)
    total = len(board)
    occupied = sum(1 for r in board if r["Статус"] == "Занято")
    return {"total": total, "occupied": occupied, "free": total - occupied}


# ==================== АНАЛИТИКА ====================

def recent_snapshots(limit: int = 20, source: str | None = None) -> list[dict]:
    if source:
        return _query(
            "SELECT id, captured_at, source, model, total, occupied, free "
            "FROM occupancy_snapshots WHERE source = :src "
            "ORDER BY captured_at DESC LIMIT :lim",
            {"src": source, "lim": limit},
        )
    return _query(
        "SELECT id, captured_at, source, model, total, occupied, free "
        "FROM occupancy_snapshots ORDER BY captured_at DESC LIMIT :lim",
        {"lim": limit},
    )


def occupancy_by_hour(source: str = "main") -> list[dict]:
    return _query("SELECT * FROM v_occupancy_by_hour WHERE source = :src", {"src": source})


def occupancy_by_weekday(source: str = "main") -> list[dict]:
    return _query("SELECT * FROM v_occupancy_by_weekday WHERE source = :src", {"src": source})


def zone_stats(source: str = "main") -> list[dict]:
    return _query("SELECT * FROM v_zone_stats WHERE source = :src", {"src": source})


# ==================== ФОРМАТИРОВАНИЕ / УТИЛИТЫ ====================

def _ru_spot(name: str) -> str:
    """Spot N -> Место N (для сайта)."""
    return name.replace("Spot", "Место")


def _fmt_dt(dt) -> str:
    if dt is None:
        return "—"
    return dt.strftime("%H:%M:%S %d.%m.%Y")


def _fmt_dur(td) -> str:
    if td is None:
        return "—"
    secs = int(td.total_seconds())
    if secs < 0:
        secs = 0
    h, rem = divmod(secs // 60, 60)
    m, s = rem, secs % 60
    if h:
        return f"{h} ч {m:02d} мин"
    if m:
        return f"{m} мин {s:02d} с"
    return f"{s} с"


def list_objects() -> list[dict]:
    """Таблицы и вьюхи в public-схеме с числом строк."""
    objects = _query(
        """
        SELECT table_name AS name,
               CASE WHEN table_type = 'VIEW' THEN 'view' ELSE 'table' END AS kind
        FROM information_schema.tables
        WHERE table_schema = 'public'
        ORDER BY kind, name
        """
    )
    for o in objects:
        try:
            o["rows"] = _query(f'SELECT COUNT(*) AS n FROM "{o["name"]}"')[0]["n"]
        except Exception:
            o["rows"] = None
    return objects


def _query(sql: str, params: dict | None = None) -> list[dict]:
    with get_engine().connect() as conn:
        return [dict(r) for r in conn.execute(text(sql), params or {}).mappings()]


def to_dataframe(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)
