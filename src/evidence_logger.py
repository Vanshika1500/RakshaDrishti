# src/evidence_logger.py
from pathlib import Path
from datetime import datetime
import uuid
import cv2

from utilities.db_utils import (
    get_db_conn,
    insert_camera,
    insert_criminal
)

EVIDENCE_DIR = Path("Data/evidence_videos")
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)


def save_video_snippet(prefix, frames, fps=20, width=None, height=None):
    if not frames:
        return None

    filename = f"{prefix}_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:6]}.mp4"
    out_path = EVIDENCE_DIR / filename

    h, w = frames[0].shape[:2]
    width = width or w
    height = height or h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    for f in frames:
        writer.write(f)
    writer.release()

    return str(out_path)


# ====================================
# FACE LOGGING
# ====================================
def log_face_recognition(camera_id, vehicle_type, number_plate,
                         person_id, person_name, confidence, video_path=None):

    with get_db_conn() as conn:

        insert_camera(conn, camera_id, vehicle_type, number_plate)

        if person_id and person_name:
            insert_criminal(conn, person_id, person_name)

        conn.execute(
            """
            INSERT INTO criminal_recognitions
            (camera_id, vehicle_type, number_plate, person_id, person_name,
             confidence, video_path, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
            """,
            (
                camera_id, vehicle_type, number_plate,
                person_id, person_name,
                float(confidence), video_path
            )
        )


# ====================================
# WEAPON LOGGING
# ====================================
def log_weapon_detection(camera_id, vehicle_type, number_plate,
                         object_type, confidence, video_path=None):

    with get_db_conn() as conn:

        insert_camera(conn, camera_id, vehicle_type, number_plate)

        conn.execute(
            """
            INSERT INTO weapon_detections
            (camera_id, vehicle_type, number_plate, object_type,
             confidence, video_path, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
            """,
            (
                camera_id, vehicle_type, number_plate,
                object_type, float(confidence), video_path
            )
        )


# ====================================
# VIOLENCE LOGGING
# ====================================
def log_violence_detection(camera_id, vehicle_type, number_plate,
                           confidence, video_path=None):

    with get_db_conn() as conn:

        insert_camera(conn, camera_id, vehicle_type, number_plate)

        conn.execute(
            """
            INSERT INTO violence_detections
            (camera_id, vehicle_type, number_plate,
             confidence, video_path, timestamp)
            VALUES (?, ?, ?, ?, ?, datetime('now'))
            """,
            (
                camera_id, vehicle_type, number_plate,
                float(confidence), video_path
            )
        )
