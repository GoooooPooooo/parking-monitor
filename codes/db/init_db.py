"""Создать/обновить схему БД и загрузить зоны.

Запуск из папки codes:
    python -m db.init_db
"""

from db import repository
from db.connection import DATABASE_URL


def main():
    print(f"Подключение: {DATABASE_URL}")
    repository.init_schema()
    print("Схема применена.")
    n = repository.sync_zones_from_json()
    print(f"Зон загружено: {n}")


if __name__ == "__main__":
    main()
