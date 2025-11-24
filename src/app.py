import streamlit as st

# -----------------------------
# MUST be the first Streamlit command
st.set_page_config(page_title="Raksha Drishti", layout="wide")
# -----------------------------

import cv2
from unified_pipeline import UnifiedPipeline

# Title
st.title("Raksha Drishti - Real-time Surveillance")

# Initialize pipeline
pipeline = UnifiedPipeline()

# Placeholders for camera feed and alerts
frame_placeholder = st.empty()
alert_placeholder = st.empty()

# Open camera
cap = cv2.VideoCapture(0)

# Edge-triggered Streamlit loop
try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Camera feed not available!")
            break

        # Process frame
        results = pipeline.process_frame(frame)

        # Show live feed
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")

        # Prepare alerts
        alerts = []
        if results["face_detected"]:
            alerts.append(f"👤 Face: {results['face_name']}")
        if results["weapon_detected"]:
            alerts.append(f"🔫 Weapon: {results['weapon_type']}")
        if results["violence_detected"]:
            alerts.append("⚠️ Violence detected!")

        # Display alerts
        if alerts:
            alert_placeholder.success(" | ".join(alerts))
        else:
            alert_placeholder.empty()

finally:
    # Release camera when Streamlit closes
    cap.release()
    cv2.destroyAllWindows()
