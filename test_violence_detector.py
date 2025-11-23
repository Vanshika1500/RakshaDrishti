import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from collections import deque

# ---------------- CONFIG ----------------
VIDEO_SOURCE = 0  # 0 = default webcam, or path to video file
MODEL_PATH = "src/weights/violence_detector.h5"
SEQUENCE_LENGTH = 16
IMAGE_HEIGHT, IMAGE_WIDTH = 96, 96
CLASSES_LIST = ["NonViolence", "Violence"]

# ---------------- LOAD MODEL ----------------
print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

# ---------------- SETUP CAMERA & BUFFER ----------------
cap = cv2.VideoCapture(VIDEO_SOURCE)
frame_buffer = deque(maxlen=SEQUENCE_LENGTH)  # rolling buffer for last 16 frames

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Preprocess frame
    resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
    normalized_frame = resized_frame / 255.0
    frame_buffer.append(normalized_frame)

    # Only predict if buffer is full
    if len(frame_buffer) == SEQUENCE_LENGTH:
        video_frames = np.expand_dims(np.array(frame_buffer), axis=0)  # shape: (1,16,96,96,3)
        pred = model.predict(video_frames, verbose=0)
        class_index = np.argmax(pred)
        confidence = pred[0][class_index]

        label = f"{CLASSES_LIST[class_index]}: {confidence*100:.2f}%"
        color = (0, 255, 0) if class_index == 0 else (0, 0, 255)  # Green for NonViolence, Red for Violence

        # Display label on frame
        cv2.putText(frame, label, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow("Real-Time Violence Detection", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

