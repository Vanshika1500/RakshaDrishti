# src/object_detection/run_object.py
import cv2
from ultralytics import YOLO
import streamlit as st

def run_object_detection(
    input_path=None,
    weights_path=None,
    conf_thresh=0.35,
    skip_frames=1,
    use_default=True,
    show_boxes=True,
    return_logs=False,
    run_flag=None
):
    model_file = "src/object_detection/detect.pt" 
    model = YOLO(model_file)

    source = 0 if input_path is None else input_path
    cap = cv2.VideoCapture(source)
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

        if frame_idx % skip_frames == 0:
            results = model(frame, conf=conf_thresh)[0]
            if show_boxes:
                frame = results.plot()

            if return_logs:
                for box in results.boxes:
                    logs.append({
                        "frame": frame_idx,
                        "class": int(box.cls.cpu().numpy()[0]),
                        "conf": float(box.conf.cpu().numpy()[0]),
                        "xyxy": box.xyxy.cpu().numpy()[0].tolist()
                    })

            # Convert BGR → RGB for Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with st.container():
                st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
                stframe.image(frame_rgb, channels="RGB", width=640)
                st.markdown("</div>", unsafe_allow_html=True)

        frame_idx += 1

    cap.release()

    if return_logs:
        return results, logs


