import sqlite3
from pathlib import Path

DB_PATH = Path("Data/database.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

CREATE_TABLES_SQL = """
PRAGMA foreign_keys = ON;

-- Cameras table (one row per camera / device)
CREATE TABLE IF NOT EXISTS cameras (
    camera_id     TEXT PRIMARY KEY,
    vehicle_type  TEXT,         -- e.g., BUS, METRO, TRAIN
    number_plate  TEXT,         -- optional
    created_at    TEXT DEFAULT (datetime('now'))
);

-- Criminals lookup table
CREATE TABLE IF NOT EXISTS criminals (
    person_id     TEXT PRIMARY KEY,
    person_name   TEXT,
    notes         TEXT,
    created_at    TEXT DEFAULT (datetime('now'))
);

-- Weapon detections
CREATE TABLE IF NOT EXISTS weapon_detections (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id         TEXT NOT NULL,
    vehicle_type      TEXT,
    number_plate      TEXT,
    object_type       TEXT,
    confidence        REAL,
    video_path        TEXT,         -- path to saved snippet
    timestamp         TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (camera_id) REFERENCES cameras(camera_id) ON DELETE SET NULL
);

-- Violence detections
CREATE TABLE IF NOT EXISTS violence_detections (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id         TEXT NOT NULL,
    vehicle_type      TEXT,
    number_plate      TEXT,
    confidence        REAL,
    video_path        TEXT,
    timestamp         TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (camera_id) REFERENCES cameras(camera_id) ON DELETE SET NULL
);

-- Criminal recognitions (face matches)
CREATE TABLE IF NOT EXISTS criminal_recognitions (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id         TEXT NOT NULL,
    vehicle_type      TEXT,
    number_plate      TEXT,
    person_id         TEXT,          -- FK to criminals.person_id (optional)
    person_name       TEXT,
    confidence        REAL,
    video_path        TEXT,
    timestamp         TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (camera_id) REFERENCES cameras(camera_id) ON DELETE SET NULL,
    FOREIGN KEY (person_id) REFERENCES criminals(person_id) ON DELETE SET NULL
);

-- Helpful indices
CREATE INDEX IF NOT EXISTS idx_weapon_camera_timestamp ON weapon_detections(camera_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_violence_camera_timestamp ON violence_detections(camera_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_criminal_camera_timestamp ON criminal_recognitions(camera_id, timestamp);
"""

def init_db(db_path=DB_PATH):
    conn = sqlite3.connect(db_path, timeout=30)
    try:
        # enable WAL for better concurrency
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.executescript(CREATE_TABLES_SQL)
        conn.commit()
        print(f"Initialized DB at {db_path}")
    finally:
        conn.close()

if __name__ == "__main__":
    init_db()
