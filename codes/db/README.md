# База данных PostgreSQL — Parking Monitor

Хранение истории заполняемости парковки для статистической обработки (по плану НИР).

## Что внутри

| Файл | Назначение |
|------|-----------|
| `schema.sql` | DDL: таблицы, индексы, вьюхи статистики |
| `connection.py` | Подключение (SQLAlchemy), параметры из переменных окружения |
| `repository.py` | Функции: схема, синхронизация зон, запись снимка, запросы статистики |
| `init_db.py` | Применить схему + загрузить зоны из `../parking_zones.json` |
| `demo.py` | Демонстрация: запись снимка и печать статистики |

## Модель данных

- **`parking_zones`** — парковочные места (координаты, тип, номер-метка).
- **`occupancy_snapshots`** — снимок состояния парковки (время, источник, модель, занято/свободно).
- **`parking_sessions`** — сессии стоянки (приезд/уезд) по каждому месту.
- **`zone_occupancy`** — занятость каждого места в рамках снимка (+ confidence).

Вьюхи для просмотра:
- **`v_parking_status`** — Состояние мест (`source | spot | status | since`);
- **`v_sessions_journal`** — Журнал посещений (`spot | arrived_at | departed_at | duration | status`);
- `v_operator_board` — подробное табло (+ последний визит);
- `v_occupancy_by_hour`, `v_occupancy_by_weekday`, `v_zone_stats` — аналитика.

> Значения во вьюхах — на английском (`Occupied`/`Free`, `Parked`/`Left`), время в формате `ЧЧ:ММ:СС ДД.ММ.ГГГГ`. Веб-интерфейс показывает те же данные на русском.

## Настройки подключения

Параметры берутся из переменных окружения (с дефолтами для локального Postgres):

```
PG_HOST=localhost
PG_PORT=5432
PG_USER=postgres
PG_PASSWORD=postgres
PG_DB=parking_monitor
```

Либо задать `DATABASE_URL=postgresql+psycopg2://user:pass@host:port/db`.
Скопируйте `.env.example` → `.env` и настройте под себя (не коммитьте пароль).

## Установка

```powershell
cd C:\parking-monitor\codes
pip install -r requirements.txt
```

## Использование

Инициализация (создать таблицы, залить зоны):

```powershell
cd C:\parking-monitor\codes
python -m db.init_db
```

Демонстрация:

```powershell
python -m db.demo
```

Из кода:

```python
from db import repository

repository.init_schema()
zones = repository.get_zones()
repository.log_snapshot(
    [{"zone_id": z["id"], "is_occupied": True, "confidence": 0.9} for z in zones],
    source="camera",
    model="yolov8x",
)
print(repository.occupancy_by_weekday())
```

## Работа из терминала (psql)

`psql` добавлен в PATH, пароль берётся из `pgpass` (не спрашивается). Для корректной
кириллицы нужен режим UTF-8 в консоли:

```powershell
chcp 65001                       # включить UTF-8 в текущей сессии
psql -U postgres -h localhost -d parking_monitor
psql -U postgres -h localhost -d parking_monitor -c "SELECT * FROM v_operator_board;"
```

Либо использовать готовую обёртку, которая сама ставит UTF-8:

```powershell
.\db\psql.cmd
.\db\psql.cmd -c "SELECT * FROM v_sessions_journal LIMIT 5;"
```

Просмотр табло и журнала без psql:

```powershell
python -m db.cli
python -m db.cli --sql "SELECT source, COUNT(*) FROM occupancy_snapshots GROUP BY source"
```

> Если вместо кириллицы видишь `?????` — консоль не в UTF-8. Выполни `chcp 65001`
> (или используй `db\psql.cmd`) и шрифт с поддержкой кириллицы (Consolas / Windows Terminal).

## Примечание

БД `parking_monitor` уже создана в локальном PostgreSQL 18. Если её нет:

```sql
CREATE DATABASE parking_monitor ENCODING 'UTF8';
```
