"""Подключение к PostgreSQL через SQLAlchemy."""

import os

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

PG_HOST = os.environ.get("PG_HOST", "localhost")
PG_PORT = os.environ.get("PG_PORT", "5432")
PG_USER = os.environ.get("PG_USER", "postgres")
PG_PASSWORD = os.environ.get("PG_PASSWORD", "postgres")
PG_DB = os.environ.get("PG_DB", "parking_monitor")

DATABASE_URL = os.environ.get("DATABASE_URL") or (
    f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"
)

_engine: Engine | None = None


def get_engine() -> Engine:
    """Вернуть singleton-движок SQLAlchemy."""
    global _engine
    if _engine is None:
        _engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)
    return _engine
