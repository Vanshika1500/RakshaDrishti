import os
import cv2
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

