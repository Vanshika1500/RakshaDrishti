# src/processors/weapon_detector.py

import os
import cv2
from ultralytics import YOLO

# Build absolute path to weights
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "weights", "finalest.pt")

print("Loading Weapon Detection Model (YOLO)...")
weapon_model = YOLO(MODEL_PATH)   # Correct path


def process_weapon(frame):
    """
    Runs YOLO weapon detection.
    Returns:
        detected (bool)
        confidence (float)
    """

    results = weapon_model(frame, verbose=False)

    detected = False
    max_conf = 0.0

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            # Assuming class 0 = weapon in your training
            if conf > max_conf:
                max_conf = conf
                detected = True

    return detected, max_conf

