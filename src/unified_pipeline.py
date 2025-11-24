import cv2
from collections import deque

from processors.face_recognition import process_face
from processors.weapon_detector import process_weapon
from processors.violence_detector import process_violence
from evidence_logger import (
    save_video_snippet,
    log_face_recognition,
    log_weapon_detection,
    log_violence_detection
)
from utilities.alert_manager import AlertManager

FPS = 20
PRE_EVENT_SEC = 5
POST_EVENT_SEC = 5
PRE_EVENT_FRAMES = FPS * PRE_EVENT_SEC
POST_EVENT_FRAMES = FPS * POST_EVENT_SEC
MIN_FRAMES = 5

class UnifiedPipeline:
    def __init__(self, camera_id="CAM01"):
        self.camera_id = camera_id

        # Buffers
        self.face_buffer = deque(maxlen=PRE_EVENT_FRAMES)
        self.weapon_buffer = deque(maxlen=PRE_EVENT_FRAMES)
        self.violence_buffer = deque(maxlen=PRE_EVENT_FRAMES)

        # Post-event counters
        self.post_event_face = 0
        self.post_event_weapon = 0
        self.post_event_violence = 0

        # Flags
        self.face_in_frame = False
        self.weapon_in_frame = False
        self.violence_in_frame = False

        # Videos to save
        self.face_video_to_save = []
        self.weapon_video_to_save = []
        self.violence_video_to_save = []

        # Alerts
        self.alert = AlertManager()

    def process_frame(self, frame):
        # Add frame to buffers
        self.face_buffer.append(frame.copy())
        self.weapon_buffer.append(frame.copy())
        self.violence_buffer.append(frame.copy())

        # ---------------- Face Detection ----------------
        face_id, face_name, face_conf, face_detected = process_face(frame)

        if face_detected and not self.face_in_frame:
            # Rising edge: new face detected
            self.face_in_frame = True
            self.post_event_face = POST_EVENT_FRAMES
            self.face_video_to_save = list(self.face_buffer)
            video_path = save_video_snippet("face", self.face_video_to_save)
            log_face_recognition(self.camera_id, "UNKNOWN", "N/A",
                                 face_id, face_name, face_conf, video_path)
            # Alert
            if face_name.lower() in ["offender", "criminal", "wanted"]:
                self.alert.high_beep()
            else:
                self.alert.low_beep()

        elif not face_detected and self.face_in_frame:
            # Falling edge: face disappeared
            self.face_in_frame = False
            self.post_event_face = POST_EVENT_FRAMES

        if self.post_event_face > 0:
            self.face_video_to_save.append(frame.copy())
            self.post_event_face -= 1

        # ---------------- Weapon Detection ----------------
        weapon_detected, weapon_conf, weapon_type = process_weapon(frame)

        if weapon_detected and not self.weapon_in_frame:
            self.weapon_in_frame = True
            self.post_event_weapon = POST_EVENT_FRAMES
            self.weapon_video_to_save = list(self.weapon_buffer)
            video_path = save_video_snippet("weapon", self.weapon_video_to_save)
            log_weapon_detection(self.camera_id, "UNKNOWN", "N/A",
                                 weapon_type, weapon_conf, video_path)
            # Alert
            self.alert.high_beep() if weapon_type.lower() != "knife" else self.alert.low_beep()

        elif not weapon_detected and self.weapon_in_frame:
            self.weapon_in_frame = False
            self.post_event_weapon = POST_EVENT_FRAMES

        if self.post_event_weapon > 0:
            self.weapon_video_to_save.append(frame.copy())
            self.post_event_weapon -= 1

        # ---------------- Violence Detection ----------------
        violence_detected, violence_conf = process_violence(frame)
        # Debug print to confirm detection
        print("Violence detected:", violence_detected, "Conf:", violence_conf)

        if violence_detected and not self.violence_in_frame:
            # Rising edge: violence starts
            self.violence_in_frame = True
            self.post_event_violence = POST_EVENT_FRAMES
            self.violence_video_to_save = list(self.violence_buffer)
            video_path = save_video_snippet("violence", self.violence_video_to_save)
            log_violence_detection(self.camera_id, "UNKNOWN", "N/A",
                                   violence_conf, video_path)
            # Alert
            if weapon_detected:
                self.alert.high_beep()
            else:
                self.alert.low_beep()

        elif not violence_detected and self.violence_in_frame:
            # Falling edge: violence ended
            self.violence_in_frame = False
            self.post_event_violence = POST_EVENT_FRAMES

        if self.post_event_violence > 0:
            self.violence_video_to_save.append(frame.copy())
            self.post_event_violence -= 1

        # ---------------- Return Detection Results ----------------
        return {
            "face_detected": face_detected,
            "face_name": face_name,
            "weapon_detected": weapon_detected,
            "weapon_type": weapon_type,
            "violence_detected": violence_detected
        }
