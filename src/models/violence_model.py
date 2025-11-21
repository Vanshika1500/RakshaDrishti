# src/models/violence_model.py
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque

MODEL_PATH = "src/weights/violence_detector.h5"
SEQUENCE_LENGTH = 16
IMAGE_HEIGHT, IMAGE_WIDTH = 96, 96

class ViolenceModel:
    def __init__(self):
        print("Loading Violence Model...")
        self.model = load_model(MODEL_PATH)
        print("Violence Model Loaded Successfully!")
        self.buffer = deque(maxlen=SEQUENCE_LENGTH)

    def predict(self, frame):
        # Resize + Normalize
        frame_resized = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
        frame_norm = frame_resized / 255.0

        # Add to buffer
        self.buffer.append(frame_norm)

        # Only predict when buffer is full
        if len(self.buffer) == SEQUENCE_LENGTH:
            input_frames = np.expand_dims(np.array(self.buffer), axis=0)
            preds = self.model.predict(input_frames, verbose=0)

            class_idx = np.argmax(preds)
            confidence = preds[0][class_idx]

            return class_idx, confidence
        return None, 0.0



def load_violence_model():
    """
    Wrapper function so unified_pipeline.py can load violence model.
     """
    return ViolenceModel()
