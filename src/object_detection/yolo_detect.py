from ultralytics import YOLO

# Load your custom trained model
model = YOLO("detect.pt")

#image_path = "test1.jpg"

# Run YOLO directly on webcam
results = model(source=0, show=True, conf=0.4, save=False)  # source=0 = webcam

"""import cv2
from ultralytics import YOLO

"import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ Camera not detected")
else:
    ret, frame = cap.read()
    if not ret:
        print("❌ Camera opened but cannot read frame")
    else:
        print("✅ Camera working, frame size:", frame.shape)

cap.release()"
def run_object_detection():
    # Load YOLO model (nano version for speed)
    model = YOLO("detect.pt")  

    # Start webcam
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection
        results = model(frame)

        # Draw detections
        annotated_frame = results[0].plot()

        # Display
        cv2.imshow("Rakshadrishti - Object Detection", annotated_frame)

        # Quit with Q key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_object_detection()"""
