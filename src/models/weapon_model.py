# src/models/weapon_model.py

from ultralytics import YOLO
from torchvision import transforms
import torch

MODEL_PATH = "weights/finalest.pt"

print("✅ Loading Weapon Detection Model (YOLO)...")
weapon_model = YOLO(MODEL_PATH)   # ← CORRECT WAY FOR YOLO MODELS

# No eval() needed — YOLO handles that internally.

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((640, 640)),  # YOLO default input size
])
