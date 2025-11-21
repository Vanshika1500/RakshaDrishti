import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

MODEL_PATH = "src/weights/violence_detector.h5"
DATASET_PATH = "Training/violence_training_data"

IMG_SIZE = 96
FRAMES_PER_CLIP = 16

model = load_model(MODEL_PATH)

def load_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame = frame.astype("float32") / 255.0
        frames.append(frame)

    cap.release()
    return frames

def make_clips(frames, clip_len=16):
    clips = []
    for i in range(0, len(frames), clip_len):
        clip = frames[i:i+clip_len]
        if len(clip) == clip_len:
            clips.append(np.array(clip))
    return clips

video_labels = []
video_preds = []

classes = ["NonViolence", "Violence"]

for label in classes:
    folder = os.path.join(DATASET_PATH, label)
    label_value = classes.index(label)

    print(f"\nProcessing folder: {label}")

    for filename in os.listdir(folder):
        if filename.endswith(".mp4"):
            video_path = os.path.join(folder, filename)

            frames = load_video_frames(video_path)
            clips = make_clips(frames, FRAMES_PER_CLIP)

            if len(clips) == 0:
                continue  # skip too-short videos

            clips = np.array(clips)

            # Predict each clip
            preds = model.predict(clips)
            clip_preds = np.argmax(preds, axis=1)

            # Majority vote for entire video
            final_pred = 1 if np.sum(clip_preds) > (len(clip_preds) / 2) else 0

            video_labels.append(label_value)
            video_preds.append(final_pred)

# Convert to numpy
y_true = np.array(video_labels)
y_pred = np.array(video_preds)

# ----- Metrics -----
print("\n===== MODEL EVALUATION =====")
print("Accuracy:", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("Recall:", recall_score(y_true, y_pred))
print("F1 Score:", f1_score(y_true, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=classes))
