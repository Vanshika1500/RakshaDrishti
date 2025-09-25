import streamlit as st
import pandas as pd
import tempfile

# Import detection functions
from src.object_detection.run_object import run_object_detection
from src.action_detection.run_pose import run_pose_detection

st.set_page_config(page_title="RakshaDrishti — Weapon & Pose", layout="wide")

st.title("🔒 RakshaDrishti — Weapon & Pose Detection")
st.caption("Real-time weapon detection (YOLOv8) + pose estimation. Use responsibly.")

with st.sidebar:
    st.header("⚙️ Controls")
    MODE = st.radio("Mode", ["📷 Webcam (real-time)", "📂 Upload video/image"], index=0)

    CONF_THRESH = st.slider("Confidence threshold", 0.1, 0.99, 0.35, 0.01)
    SKIP_FRAMES = st.slider("Process every Nth frame", 1, 10, 2)

    SHOW_BOXES = st.checkbox("Show bounding boxes", value=True)
    SHOW_POSE = st.checkbox("Show pose skeleton", value=True)

    uploaded_weights = st.file_uploader("Upload custom YOLO weights (.pt)", type=["pt"])
    use_default_weights = st.checkbox("Use default weights (yolov8n.pt)", value=True)

    st.markdown("---")
    st.info("💡 Tip: For best performance, run on a machine with GPU.")



def show_live_webcam():
    st.subheader("📷 Live Webcam Detection")

    col1, col2 = st.columns(2)
    with col1:
        st.button("▶️ Start", on_click=start_detection)
    with col2:
        st.button("⏹ Stop", on_click=stop_detection)

    if st.session_state.run_detection:
        weights_path = None
        if uploaded_weights is not None:
            t = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
            t.write(uploaded_weights.read())
            t.flush()
            weights_path = t.name

        if SHOW_POSE:
            from src.utils.run_combined import run_combined_detection
            run_combined_detection(
                source="webcam",
                weights_path=weights_path or "Models/finalest.pt",
                conf_obj=CONF_THRESH,
                conf_pose=0.25,
                skip_frames=SKIP_FRAMES,
                run_flag=lambda: st.session_state.run_detection
            )
        else:
            run_object_detection(
                weights_path=weights_path,
                conf_thresh=CONF_THRESH,
                skip_frames=SKIP_FRAMES,
                use_default=use_default_weights,
                run_flag=lambda: st.session_state.run_detection
            )


def show_upload_processor():
    st.subheader("📂 Process Uploaded Video/Image")
    uploaded = st.file_uploader("Upload video or image", type=["mp4", "mov", "avi", "mkv", "jpg", "jpeg", "png"])
    if not uploaded:
        st.info("Upload a file to start processing.")
        return

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded.read())
    tfile.flush()
    path = tfile.name

    st.success("File uploaded. Click below to start processing.")

    if st.button("▶️ Start Processing"):
        results, logs = run_object_detection(
            input_path=path,
            conf_thresh=CONF_THRESH,
            show_boxes=SHOW_BOXES,
            skip_frames=SKIP_FRAMES,
            use_default=use_default_weights,
            return_logs=True
        )

        if SHOW_POSE:
            from src.utils.run_combined import run_combined_detection
            run_combined_detection(
            source="webcam",
            weights_path="Models/yolov8n-pose.pt",
            conf_obj=CONF_THRESH,
            conf_pose=0.25,
            skip_frames=SKIP_FRAMES,
            run_flag=lambda: st.session_state.run_detection
            )
        else:
            run_object_detection(
                weights_path=weights_path,
                conf_thresh=CONF_THRESH,
                skip_frames=SKIP_FRAMES,
                use_default=use_default_weights,
                run_flag=lambda: st.session_state.run_detection
            )



if "run_detection" not in st.session_state:
    st.session_state.run_detection = False

def start_detection():
    st.session_state.run_detection = True

def stop_detection():
    st.session_state.run_detection = False


if MODE == "📷 Webcam (real-time)":
    show_live_webcam()
else:
    show_upload_processor()

st.markdown("---")
st.caption("⚠️ Disclaimer: This is a research prototype. False positives/negatives may occur. Use responsibly.")
