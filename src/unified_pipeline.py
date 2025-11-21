# unified_pipeline.py
import cv2
from collections import deque

# --- Import processors ---
from src.processors.face_recognition import process_face
from src.processors.weapon_detector import process_weapon
from src.processors.violence_detector import process_violence


# --- Import logging ---
from src.evidence_logger import (
    save_video_snippet,
    log_face_recognition,
    log_weapon_detection,
    log_violence_detection
)


def run_unified_pipeline(
    camera_id="CAM01",
    vehicle_type="UNKNOWN",
    number_plate="N/A",
    source=0  # webcam default
):
    """
    Runs real-time unified video pipeline:
    - Face Recognition
    - Weapon Detection
    - Violence Detection
    """

    cap = cv2.VideoCapture(source)
    buffer = deque(maxlen=40)  # stores last N frames for evidence video

    # cooldowns prevent multiple logs every frame
    cooldown_face = 0
    cooldown_weapon = 0
    cooldown_violence = 0

    print("Unified pipeline running... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        buffer.append(frame.copy())

        # -------------------------------
        # 1️⃣ FACE RECOGNITION
        # -------------------------------
        face_id, face_name, face_confidence, face_detected = process_face(frame)

        if face_detected and cooldown_face == 0:
            video_path = save_video_snippet("face", list(buffer))
            log_face_recognition(
                camera_id,
                vehicle_type,
                number_plate,
                face_id,
                face_name,
                face_confidence,
                video_path
            )
            cooldown_face = 50


        # -------------------------------
        # 2️⃣ WEAPON DETECTION
        # -------------------------------
        weapon_detected, weapon_confidence = process_weapon(frame)

        if weapon_detected and cooldown_weapon == 0:
            video_path = save_video_snippet("weapon", list(buffer))
            log_weapon_detection(
                camera_id,
                vehicle_type,
                number_plate,
                "weapon",
                weapon_confidence,
                video_path
            )
            cooldown_weapon = 50


        # -------------------------------
        # 3️⃣ VIOLENCE DETECTION
        # -------------------------------
        violence_detected, violence_confidence = process_violence(frame)

        if violence_detected and cooldown_violence == 0:
            video_path = save_video_snippet("violence", list(buffer))
            log_violence_detection(
                camera_id,
                vehicle_type,
                number_plate,
                "violence",
                violence_confidence,
                video_path
            )
            cooldown_violence = 50


        # -------------------------------
        # VISUAL FEEDBACK ON FRAME
        # -------------------------------
        cv2.putText(frame, f"Face Detected: {face_name}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.putText(frame, f"Weapon: {weapon_confidence:.2f}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255) if weapon_detected else (0, 255, 0), 2)

        cv2.putText(frame, f"Violence: {violence_confidence:.2f}", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255) if violence_detected else (0, 255, 0), 2)

        cv2.imshow("Raksha Drishti Unified Pipeline", frame)


        # update cooldowns
        cooldown_face = max(0, cooldown_face - 1)
        cooldown_weapon = max(0, cooldown_weapon - 1)
        cooldown_violence = max(0, cooldown_violence - 1)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_unified_pipeline()
