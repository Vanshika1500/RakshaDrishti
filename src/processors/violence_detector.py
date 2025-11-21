# src/processors/violence_detector.py

import cv2
from src.models.violence_model import ViolenceModel

# load model once globally
violence_model = ViolenceModel()

def process_violence(frame):
    """
    Returns:
        violence_detected (bool)
        confidence (float)
    """
    class_idx, confidence = violence_model.predict(frame)

    if class_idx is None:
        return False, 0.0

    # Class index: 0 = NonViolence, 1 = Violence
    violence_detected = (class_idx == 1)

    return violence_detected, confidence
