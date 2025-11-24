# src/processors/weapon_detector.py

import os
import cv2
from ultralytics import YOLO

# Build absolute path to weights
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "weights", "weapon.pt")

print("Loading Weapon Detection Model (YOLO)...")
weapon_model = YOLO(MODEL_PATH)   # Correct path

# Map YOLO class IDs to weapon types (update according to your training)
CLASS_MAP = {
    0: "gun",
    1: "knife",
    # add more if needed
}

def process_weapon(frame):
    """
    Runs YOLO weapon detection.
    Returns:
        detected (bool)
        confidence (float)
        weapon_type (str) -> 'knife', 'gun', or 'unknown'
    """

    results = weapon_model(frame, verbose=False)

    detected = False
    max_conf = 0.0
    weapon_type = "unknown"

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            # check max confidence
            if conf > max_conf:
                max_conf = conf
                detected = True
                weapon_type = CLASS_MAP.get(cls, "unknown")

    return detected, max_conf, weapon_type
