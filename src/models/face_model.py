# src/models/face_model.py

import cv2
from keras_facenet import FaceNet

print(" Loading FaceNet model")
#embedder = FaceNet()

embedder = FaceNet(cache_folder="D:/keras_cache") 


def extract_faces(img_bgr):
    """
    Extract face boxes + embeddings from BGR frame.
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return embedder.extract(img_rgb, threshold=0.95)

def get_embedding_from_path(image_path):
    """
    Reads an image path, detects a face, and returns its embedding.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Cannot read image: {image_path}")
        return None

    detections = extract_faces(img)

    if len(detections) == 0:
        print(f"[WARN] No face detected in: {image_path}")
        return None

    return detections[0]["embedding"]
