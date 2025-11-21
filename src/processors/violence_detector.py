from src.models.violence_model import load_violence_model
from src.processors.violence_detection import detect_violence

violence_model = load_violence_model()

def process_violence(frame):
    detected, score = detect_violence(frame, violence_model)
    return detected, score
