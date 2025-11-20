# db_utils.py
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

def insert_camera(camera_id, vehicle_type=None, number_plate=None):
    with get_db_conn() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO cameras (camera_id, vehicle_type, number_plate) VALUES (?, ?, ?)",
            (camera_id, vehicle_type, number_plate)
        )

def insert_criminal(person_id, person_name, notes=None):
    with get_db_conn() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO criminals (person_id, person_name, notes) VALUES (?, ?, ?)",
            (person_id, person_name, notes)
        )

# snippet buffer
from collections import deque
import time

class FrameBuffer:
    def __init__(self, maxlen=150):
        # maxlen roughly = fps * seconds_to_keep (150 ~ 7.5s at 20fps)
        self.buffer = deque(maxlen=maxlen)

    def add_frame(self, frame):
        # frame is an OpenCV image (numpy array)
        self.buffer.append((time.time(), frame.copy()))

    def get_frames(self, pre_seconds=3, post_frames=30):
        # returns list of frames: last pre_seconds worth + placeholder for post_frames
        return [f for (_, f) in list(self.buffer)]

    def clear(self):
        self.buffer.clear()
