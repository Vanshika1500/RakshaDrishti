from src.db_utils import get_db_conn, insert_camera, insert_criminal
from pathlib import Path
import uuid
import cv2
from datetime import datetime

# Directory to save evidence video clips
EVIDENCE_DIR = Path("Data/evidence_videos")
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)


def save_video_snippet(frames, fps=20, width=None, height=None):
    """
    Saves a short snippet of video frames (e.g., when a criminal is recognized).
    frames: list/iterable of BGR numpy frames (OpenCV)
    Returns the file path of the saved video.
    """
    if not frames:
        return None

    filename = f"criminal_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:6]}.mp4"
    out_path = EVIDENCE_DIR / filename

    # derive frame size from first frame
    h, w = frames[0].shape[:2]
    if width is None: width = w
    if height is None: height = h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    for f in frames:
        writer.write(f)
    writer.release()

    return str(out_path)


def log_criminal_recognition(camera_id, vehicle_type, number_plate, person_id, person_name, confidence, video_path=None):
    """
    Logs recognized criminal details into the database.
    - camera_id: ID of the CCTV camera
    - vehicle_type: type of vehicle detected
    - number_plate: vehicle plate number
    - person_id: unique criminal ID (if known)
    - person_name: name of recognized criminal
    - confidence: recognition confidence score
    - video_path: path to saved evidence video snippet
    """
    insert_camera(camera_id, vehicle_type, number_plate)
    if person_id and person_name:
        insert_criminal(person_id, person_name)

    with get_db_conn() as conn:
        conn.execute(
            """INSERT INTO criminal_recognitions
               (camera_id, vehicle_type, number_plate, person_id, person_name, confidence, video_path, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))""",
            (camera_id, vehicle_type, number_plate, person_id, person_name, float(confidence), video_path)
        )


# Example usage:
# frames = deque or list of frames captured during recognition
# video_path = save_video_snippet(frames)
# log_criminal_recognition("CAM01", "BUS", "MH12AB1234", "P023", "John Doe", 0.95, video_path)

