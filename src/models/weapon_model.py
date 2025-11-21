# src/models/weapon_model.py

from ultralytics import YOLO
import os

# Correct model path
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "weights", "finalest.pt")

print("✅ Loading Weapon Detection Model (YOLO)...")

# Load model only once
weapon_model = YOLO(MODEL_PATH)
