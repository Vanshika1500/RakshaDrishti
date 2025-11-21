from tensorflow.keras.models import load_model

INPUT_PATH = "weights/violence_detector.keras"
OUTPUT_PATH = "weights/violence_detector.h5"

print("Loading model:", INPUT_PATH)
model = load_model(INPUT_PATH)

print("Saving as H5:", OUTPUT_PATH)
model.save(OUTPUT_PATH)

print("\n🎉 Done! violence_detector.h5 generated successfully.")

