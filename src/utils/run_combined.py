# src/utils/run_combined.py
import cv2
from ultralytics import YOLO
import streamlit as st

def run_combined_detection(
    source=0,
    weights_path="src/object_detection/detect.pt",
    conf_obj=0.35,
    conf_pose=0.25,
    imgsz_pose=480,
    skip_frames=1,
    run_flag=lambda: True
):
    """
    Run object detection + pose estimation together on webcam/video.
    """
    # Load models
    obj_model = YOLO(weights_path)
    pose_model = YOLO("src/action_detection/yolov8n-pose.pt")   # ✅ use standard pose model

    cap = cv2.VideoCapture(0 if source == "webcam" else source)
    if not cap.isOpened():
        raise RuntimeError(f"❌ Cannot open source: {source}")

    stframe = st.empty()
    frame_idx = 0

    while run_flag() and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if frame_idx % skip_frames == 0:
            raw_frame = frame.copy()

            # Run object detection
            obj_results = obj_model(raw_frame, conf=conf_obj)[0]
            frame_with_boxes = obj_results.plot()

            # Run pose detection (on raw frame, not already drawn frame)
            pose_results = pose_model(raw_frame, conf=conf_pose, imgsz=imgsz_pose)[0]
            frame_with_pose = pose_results.plot()

            # Blend both
            combined = cv2.addWeighted(frame_with_boxes, 0.7, frame_with_pose, 0.7, 0)

            # Show in Streamlit
            frame_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
            with st.container():
                st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
                stframe.image(frame_rgb, channels="RGB", width=640)
                st.markdown("</div>", unsafe_allow_html=True)

    cap.release()

