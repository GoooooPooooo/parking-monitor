"""Демонстрация работы с БД: запись снимка и вывод статистики.

Запуск из папки codes:
    python -m db.demo
"""

from db import repository


def main():
    repository.init_schema()
    n = repository.sync_zones_from_json()
    print(f"Зон синхронизировано: {n}")

    zones = repository.get_zones()
    states = [
        {"zone_id": z["id"], "is_occupied": (i % 2 == 0), "confidence": 0.9}
        for i, z in enumerate(zones)
    ]
    sid = repository.log_snapshot(states, source="demo", model="yolov8x")
    print(f"Записан снимок id={sid}: занято {sum(s['is_occupied'] for s in states)}/{len(states)}")

    print("\nПоследние снимки:")
    for r in repository.recent_snapshots(5):
        print("  ", r)

    print("\nЗанятость по часам:")
    for r in repository.occupancy_by_hour():
        print("  ", r)

    print("\nСтатистика по местам:")
    for r in repository.zone_stats():
        print("  ", r)


if __name__ == "__main__":
    main()
