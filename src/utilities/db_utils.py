# src/utilities/db_utils.py
import sqlite3
from pathlib import Path
from contextlib import contextmanager

DB_PATH = Path("Data/database.db")


@contextmanager
def get_db_conn():
    conn = sqlite3.connect(DB_PATH, timeout=30, detect_types=sqlite3.PARSE_DECLTYPES)
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        yield conn
    finally:
        conn.commit()
        conn.close()


def insert_camera(conn, camera_id, vehicle_type=None, number_plate=None):
    conn.execute(
        """
        INSERT OR IGNORE INTO cameras
        (camera_id, vehicle_type, number_plate)
        VALUES (?, ?, ?)
        """,
        (camera_id, vehicle_type, number_plate)
    )


def insert_criminal(conn, person_id, person_name, notes=None):
    conn.execute(
        """
        INSERT OR IGNORE INTO criminals
        (person_id, person_name, notes)
        VALUES (?, ?, ?)
        """,
        (person_id, person_name, notes)
    )
