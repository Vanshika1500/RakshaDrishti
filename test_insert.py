# test_insert.py
from Models.weapon_detection import log_weapon_detection
from Models.violence_detection import log_violence_detection
from Models.facial_recognition import log_criminal_recognition

# simulate
log_weapon_detection("CAM01", "BUS", "MH12AB1234", "knife", 0.92, "Data/evidence_videos/sample_weapon.mp4")
log_violence_detection("CAM01", "BUS", "MH12AB1234", 0.89, "Data/evidence_videos/sample_violence.mp4")
log_criminal_recognition("CAM01", "BUS", "MH12AB1234", "P007", "Raman Singh", 0.96, "Data/evidence_videos/sample_face.mp4")
print("Inserted sample rows.")
