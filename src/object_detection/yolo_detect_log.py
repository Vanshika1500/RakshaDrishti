from ultralytics import YOLO

# Load your custom trained model
model = YOLO("detect.pt")

#image_path = "test1.jpg"

# Run YOLO directly on webcam
results = model(source=0, show=True, conf=0.4, save=False)  # source=0 = webcam


