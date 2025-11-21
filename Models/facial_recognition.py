from src.db_utils import get_db_conn, insert_camera, insert_criminal
from pathlib import Path
import uuid
import cv2
from datetime import datetime
import os
import numpy as np
from keras_facenet import FaceNet
from tensorflow.keras.preprocessing import image
from numpy.linalg import norm


# -----------------------------
# STEP 1: Load pretrained FaceNet model
# -----------------------------
embedder = FaceNet()

print(" Face recognition script started")

# -----------------------------
# STEP 2: Function to preprocess and get embeddings
# -----------------------------
def get_embedding(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f" Could not load {img_path}")
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = embedder.extract(img, threshold=0.95)
    if len(faces) == 0:
        print(f"No face detected in {img_path}")
        return None
    embedding = faces[0]['embedding']
    return embedding

# -----------------------------
# STEP 3: Load known offenders
# -----------------------------
def load_offender_database(folder='offenders'):
    db = {}
    for person in os.listdir(folder):
        person_folder = os.path.join(folder, person)
        if not os.path.isdir(person_folder):
            continue
        embeddings = []
        for img_file in os.listdir(person_folder):
            path = os.path.join(person_folder, img_file)
            emb = get_embedding(path)
            if emb is not None:
                embeddings.append(emb)
        if embeddings:
            db[person] = np.mean(embeddings, axis=0)
            print(f" Loaded {person} with {len(embeddings)} embeddings")
    return db

offender_db = load_offender_database("offenders")

# -----------------------------
# STEP 4: Compare embeddings and draw boxes
# -----------------------------
def recognize_face(frame):
    results = embedder.extract(frame, threshold=0.95)
    if len(results) == 0:
        return frame

    for res in results:
        # FIX: convert (x, y, w, h) → (x1, y1, x2, y2)
        x, y, w, h = res['box']
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)

        embedding = res['embedding']

        # Compare with offenders
        min_dist = 1.0
        identity = "Unknown"
        for name, db_emb in offender_db.items():
            dist = norm(embedding - db_emb)
            if dist < min_dist and dist < 0.9:  # threshold (lower = stricter)
                min_dist = dist
                identity = name

        color = (0, 0, 255) if identity != "Unknown" else (0, 255, 0)
        label = f"{identity}" if identity == "Unknown" else f"ALERT: {identity}"

        # Draw rectangle & label
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return frame

# -----------------------------
# STEP 5: Real-time webcam / image demo
# -----------------------------
def run_demo(use_webcam=True, test_image=None):
    if use_webcam:
        cap = cv2.VideoCapture(0)
        print(" Webcam started. Press 'q' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            result = recognize_face(frame)
            cv2.imshow("Raksha Drishti - FRS Demo", result)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        img = cv2.imread(test_image)
        result = recognize_face(img)
        cv2.imshow("Test Image Result", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# -----------------------------
# STEP 6: Run
# -----------------------------
if __name__ == "__main__":
    print(" FRS script started")
    print("Initializing demo...")
    run_demo(use_webcam=True)
    print(" Script ended")




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

