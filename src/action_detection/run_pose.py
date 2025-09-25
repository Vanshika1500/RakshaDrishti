# src/action_detection/run_pose.py
import time, json, os
import cv2
import numpy as np
from ultralytics import YOLO
import streamlit as st

def norm_kps(kps, frame_w, frame_h):
    """Normalize YOLOv8 keypoints safely for any version"""
    norm = []
    for joint in kps:
        try:
            if len(joint) == 3:
                x, y, c = joint
            elif len(joint) == 2:
                x, y = joint
                c = 1.0
            else:
                x = y = c = 0.0
        except Exception:
            x = y = c = 0.0

        if x is None or y is None or np.isnan(x) or np.isnan(y):
            x = y = c = 0.0

        norm.append([float(x)/frame_w, float(y)/frame_h, float(c)])

    return norm


def run_pose_detection(
    source=0,
    imgsz=320,
    conf=0.25,
    show=True,
    label=None,
    log_path=None,
    run_flag=lambda: True
):
    """
    Run YOLOv8 pose detection inside Streamlit.

    Args:
        source (str|int): Webcam index, "webcam", or file path.
        imgsz (int): Image size.
        conf (float): Confidence threshold.
        show (bool): Show preview inside Streamlit.
        label (str|None): Optional label to log.
        log_path (str|None): Path to save JSON logs.
        run_flag (callable): Function that returns True/False to control loop.

    Returns:
        logs (list): Pose detection logs.
    """
    model = YOLO("src/action_detection/yolov8n-pose.pt")
    cap = cv2.VideoCapture(0 if source == "webcam" else source)

    if not cap.isOpened():
        raise RuntimeError(f"❌ Cannot open source: {source}")

    logs = []
    frame_idx = 0

    # Streamlit placeholder for frames
    stframe = st.empty()

    while run_flag() and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        H, W = frame.shape[:2]

        results = model(frame, imgsz=imgsz, conf=conf)[0]

        persons = []
        try:
            kps_all = results.keypoints.cpu().numpy()
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
        except Exception:
            kps_all, boxes, confs = None, None, None

        if kps_all is not None:
            for i, kp in enumerate(kps_all):
                persons.append({
                    "bbox": boxes[i].tolist() if boxes is not None else None,
                    "keypoints": norm_kps(kp, W, H),
                    "box_conf": float(confs[i]) if confs is not None else None
                })

        log = {
            "t": time.time(),
            "frame_idx": frame_idx,
            "label": label,
            "persons": persons,
            "frame_size": [W, H]
        }
        logs.append(log)

        if log_path:
            os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
            with open(log_path, "a") as f:
                f.write(json.dumps(log) + "\n")

        if show:
            vis = results.plot()
            vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            with st.container():
                st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
                stframe.image(frame_rgb, channels="RGB", width=640)
                st.markdown("</div>", unsafe_allow_html=True)

    cap.release()
    return logs

