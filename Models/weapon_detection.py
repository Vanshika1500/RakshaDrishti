from src.db_utils import get_db_conn, insert_camera
from pathlib import Path
import uuid
import os
from datetime import datetime

EVIDENCE_DIR = Path("Data/evidence_videos")
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

def save_video_snippet(frames, fps=20, width=None, height=None):
    """
    frames: list/iterable of BGR numpy frames (OpenCV)
    Returns path to saved file.
    """
    import cv2
    if not frames:
        return None
    filename = f"weapon_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:6]}.mp4"
    out_path = EVIDENCE_DIR / filename

    # derive size from first frame if not provided
    h, w = frames[0].shape[:2]
    if width is None: width = w
    if height is None: height = h

    # use MP4 container and H264 codec if available
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    for f in frames:
        writer.write(f)
    writer.release()
    return str(out_path)

def log_weapon_detection(camera_id, vehicle_type, number_plate, object_type, confidence, video_path=None):
    insert_camera(camera_id, vehicle_type, number_plate)
    with get_db_conn() as conn:
        conn.execute(
            """INSERT INTO weapon_detections
                (camera_id, vehicle_type, number_plate, object_type, confidence, video_path, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, datetime('now'))""",
            (camera_id, vehicle_type, number_plate, object_type, float(confidence), video_path)
        )

# Example usage when detection happens
# frames = deque buffer frames -> pass to save_video_snippet(frames)
# video_path = save_video_snippet(frames)
# log_weapon_detection("CAM01", "BUS", "MH12AB1234", "knife", 0.92, video_path)
