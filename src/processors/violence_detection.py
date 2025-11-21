import cv2
import numpy as np

def preprocess_frame(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

def detect_violence(frame, model, threshold=0.6):
    inp = preprocess_frame(frame)
    prob = float(model.predict(inp)[0][0])
    return prob > threshold, prob
