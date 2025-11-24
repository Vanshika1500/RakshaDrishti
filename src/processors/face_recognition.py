# src/processors/face_recognition.py

import cv2
import numpy as np
from numpy.linalg import norm
from models.face_model import extract_faces
from offender_database import load_offender_database

# Load offender embeddings once
offender_db = load_offender_database()


# ----------------------------------------------------------
# OPTIONAL: Function to draw recognition results on frame
# ----------------------------------------------------------
def recognize_faces(frame):
    faces = extract_faces(frame)

    if len(faces) == 0:
        return frame

    for res in faces:
        x, y, w, h = res["box"]
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)

        embedding = res["embedding"]

        identity = "Unknown"
        min_dist = 1.0

        for name, db_emb in offender_db.items():
            dist = norm(embedding - db_emb)
            if dist < min_dist and dist < 0.9:
                min_dist = dist
                identity = name

        # drawing
        color = (0, 0, 255) if identity != "Unknown" else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, identity, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return frame



# ----------------------------------------------------------
# MAIN FUNCTION USED BY UNIFIED PIPELINE
# Returns:
#   person_id, person_name, confidence, detected_flag
# ----------------------------------------------------------
def process_face(frame):

    faces = extract_faces(frame)

    if len(faces) == 0:
        return None, None, 0.0, False

    # For now we only process 1 face (first detection)
    res = faces[0]

    x, y, w, h = res["box"]
    embedding = res["embedding"]

    best_name = "Unknown"
    best_dist = 999

    # Compare to offender database
    for name, db_emb in offender_db.items():
        dist = norm(embedding - db_emb)
        if dist < best_dist:
            best_dist = dist
            best_name = name

    # FaceNet threshold for match
    MATCH_THRESHOLD = 0.90

    if best_dist < MATCH_THRESHOLD:
        # Known criminal detected
        return best_name, best_name, float(best_dist), True
    else:
        # Unknown person
        return None, "Unknown", float(best_dist), True
