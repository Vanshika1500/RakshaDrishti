# src/processors/weapon_detector.py

import cv2
from ultralytics import YOLO

# Load YOLO model just once
print("Loading Weapon Detection Model (YOLO)...")
weapon_model = YOLO("weights/finalest.pt")   # Your YOLO model path


def process_weapon(frame):
    """
    Runs YOLO weapon detection.
    Returns:
        detected (bool)
        confidence (float)
    """

    # Run YOLO inference
    results = weapon_model(frame, verbose=False)

    detected = False
    max_conf = 0.0

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            # Weapon class check (cls==0 for most custom models)
            if conf > max_conf:
                max_conf = conf
                detected = True

    return detected, max_conf
