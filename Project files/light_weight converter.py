import tensorflow as tf # type: ignore
from tensorflow.keras.models import load_model# type: ignore

# Load your original .h5 model
model = load_model("vgg16_model.h5")

# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optional: Quantize to make it even smaller
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert the model
tflite_model = converter.convert()

# Save the model
with open("vgg16_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Converted to TensorFlow Lite and saved!")




