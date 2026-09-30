-- Схема БД мониторинга заполненности парковки (PostgreSQL 18)
-- Идемпотентно: можно применять повторно.

DROP VIEW IF EXISTS v_parking_status;
DROP VIEW IF EXISTS v_operator_board;
DROP VIEW IF EXISTS v_sessions_journal;
DROP VIEW IF EXISTS v_occupancy_by_hour;
DROP VIEW IF EXISTS v_occupancy_by_weekday;
DROP VIEW IF EXISTS v_zone_stats;

CREATE TABLE IF NOT EXISTS parking_zones (
    id          SERIAL      PRIMARY KEY,
    source      TEXT        NOT NULL DEFAULT 'main',
    name        TEXT        NOT NULL,
    zone_type   TEXT        NOT NULL DEFAULT 'poly',
    points      JSONB,
    label_x     INTEGER,
    label_y     INTEGER,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    is_active   BOOLEAN     NOT NULL DEFAULT TRUE,
    UNIQUE (source, name)
);

CREATE TABLE IF NOT EXISTS occupancy_snapshots (
    id          BIGSERIAL   PRIMARY KEY,
    captured_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    source      TEXT        NOT NULL DEFAULT 'main',
    model       TEXT,
    total       INTEGER     NOT NULL DEFAULT 0,
    occupied    INTEGER     NOT NULL DEFAULT 0,
    free        INTEGER     NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS zone_occupancy (
    id           BIGSERIAL   PRIMARY KEY,
    snapshot_id  BIGINT      NOT NULL REFERENCES occupancy_snapshots(id) ON DELETE CASCADE,
    zone_id      INTEGER     NOT NULL REFERENCES parking_zones(id) ON DELETE CASCADE,
    is_occupied  BOOLEAN     NOT NULL,
    confidence   REAL,
    captured_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Сессии стоянки: приезд / уезд по каждому месту
CREATE TABLE IF NOT EXISTS parking_sessions (
    id          BIGSERIAL   PRIMARY KEY,
    zone_id     INTEGER     NOT NULL REFERENCES parking_zones(id) ON DELETE CASCADE,
    arrived_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    departed_at TIMESTAMPTZ,
    is_open     BOOLEAN     NOT NULL DEFAULT TRUE
);

CREATE INDEX IF NOT EXISTS idx_zone_occ_captured   ON zone_occupancy (captured_at);
CREATE INDEX IF NOT EXISTS idx_zone_occ_zone       ON zone_occupancy (zone_id);
CREATE INDEX IF NOT EXISTS idx_zone_occ_snapshot   ON zone_occupancy (snapshot_id);
CREATE INDEX IF NOT EXISTS idx_snapshots_captured  ON occupancy_snapshots (captured_at);
CREATE INDEX IF NOT EXISTS idx_sessions_zone_open  ON parking_sessions (zone_id, is_open);
CREATE INDEX IF NOT EXISTS idx_sessions_arrived    ON parking_sessions (arrived_at);

-- СОСТОЯНИЕ МЕСТ: компактно — место и текущий статус
CREATE OR REPLACE VIEW v_parking_status AS
SELECT
    z.source,
    z.name AS spot,
    CASE WHEN open_s.id IS NOT NULL THEN 'Occupied' ELSE 'Free' END AS status,
    to_char(open_s.arrived_at, 'HH24:MI:SS DD.MM.YYYY') AS since
FROM parking_zones z
LEFT JOIN LATERAL (
    SELECT s.id, s.arrived_at FROM parking_sessions s
    WHERE s.zone_id = z.id AND s.is_open
    ORDER BY s.arrived_at DESC LIMIT 1
) open_s ON TRUE
WHERE z.is_active
ORDER BY z.id;

-- ТАБЛО ОПЕРАТОРА: текущее состояние каждого места + последний визит
CREATE OR REPLACE VIEW v_operator_board AS
SELECT
    z.id                             AS zone_id,
    z.source                         AS source,
    z.name                           AS spot,
    CASE WHEN open_s.id IS NOT NULL THEN 'Occupied' ELSE 'Free' END AS status,
    to_char(open_s.arrived_at,  'HH24:MI:SS DD.MM.YYYY') AS arrived_at,
    to_char(last_s.arrived_at,  'HH24:MI:SS DD.MM.YYYY') AS last_arrived_at,
    to_char(last_s.departed_at, 'HH24:MI:SS DD.MM.YYYY') AS last_departed_at
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
WHERE z.is_active
ORDER BY z.id;

-- ЖУРНАЛ ПОСЕЩЕНИЙ: сессии стоянок
CREATE OR REPLACE VIEW v_sessions_journal AS
SELECT
    s.id,
    z.source                         AS source,
    z.name                           AS spot,
    to_char(s.arrived_at,  'HH24:MI:SS DD.MM.YYYY') AS arrived_at,
    to_char(s.departed_at, 'HH24:MI:SS DD.MM.YYYY') AS departed_at,
    CASE WHEN s.is_open THEN 'Parked' ELSE 'Left' END AS status,
    to_char(COALESCE(s.departed_at, now()) - s.arrived_at, 'HH24:MI:SS') AS duration
FROM parking_sessions s
JOIN parking_zones z ON z.id = s.zone_id
ORDER BY s.arrived_at DESC;

-- Аналитика: средняя занятость по часам суток
CREATE OR REPLACE VIEW v_occupancy_by_hour AS
SELECT
    source,
    EXTRACT(HOUR FROM captured_at)::int                                         AS hour_of_day,
    COUNT(*)                                                                    AS snapshots,
    ROUND(AVG(occupied)::numeric, 2)                                            AS avg_occupied,
    ROUND(AVG(free)::numeric, 2)                                               AS avg_free,
    ROUND(AVG(CASE WHEN total > 0 THEN occupied::numeric / total END) * 100, 1) AS avg_fill_percent
FROM occupancy_snapshots
GROUP BY source, 2
ORDER BY source, 2;

-- Аналитика: по дням недели (1 = Пн ... 7 = Вс)
CREATE OR REPLACE VIEW v_occupancy_by_weekday AS
SELECT
    source,
    EXTRACT(ISODOW FROM captured_at)::int                                       AS weekday,
    TO_CHAR(captured_at, 'Dy')                                                  AS weekday_name,
    COUNT(*)                                                                    AS snapshots,
    ROUND(AVG(occupied)::numeric, 2)                                           AS avg_occupied,
    ROUND(AVG(CASE WHEN total > 0 THEN occupied::numeric / total END) * 100, 1) AS avg_fill_percent
FROM occupancy_snapshots
GROUP BY source, 2, 3
ORDER BY source, 2;

-- Аналитика: по каждому месту
CREATE OR REPLACE VIEW v_zone_stats AS
SELECT
    z.source,
    z.id,
    z.name,
    COUNT(zo.id)                                                     AS measurements,
    ROUND(AVG(CASE WHEN zo.is_occupied THEN 1 ELSE 0 END) * 100, 1) AS occupied_percent
FROM parking_zones z
LEFT JOIN zone_occupancy zo ON zo.zone_id = z.id
GROUP BY z.source, z.id, z.name
ORDER BY z.source, z.id;
