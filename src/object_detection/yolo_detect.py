import cv2
from ultralytics import YOLO

def run_object_detection():
    # Load YOLO model (nano version for speed)
    model = YOLO("yolov8n.pt")  

    # Start webcam
    cap = cv2.VideoCapture(0)

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
    run_object_detection()
