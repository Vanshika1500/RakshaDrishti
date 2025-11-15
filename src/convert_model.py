import os
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

old_model_path = os.path.join(BASE_DIR, "weights", "violence_detector.keras")
new_model_path = os.path.join(BASE_DIR, "weights", "violence_detector_fixed.keras")

print("Loading:", old_model_path)

model = tf.keras.models.load_model(old_model_path, compile=False)

print("Saving new model:", new_model_path)
model.save(new_model_path, save_format="keras")

print("DONE — Model successfully converted!")
