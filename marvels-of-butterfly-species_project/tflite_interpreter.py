import tensorflow as tf # type: ignore
import numpy as np# type: ignore
from PIL import Image# type: ignore
import os

# Mapping class index to butterfly names
butterfly_names = {
    0: 'ADONIS', 1: 'AFRICAN GIANT SWALLOWTAIL', 2: 'AMERICAN SHOOT',
    3: 'AN HR', 4: 'APPOLLO', 5: 'ARIAIA', 6: 'BANDED ORANGE HELICONTAN',
    7: 'BANDED PEACOCK', 8: 'BECKERS WHITE', 9: 'BLACK HAIRSTREAK',
    # ... add all up to 74
    74: 'ZEBRA LONG WING'
}

# ✅ Define your image path
file_path = r"C:\Users\reddy\Downloads\butterflies\Butterfly Identification\test\Image_2.jpg"

# ✅ Define TFLite model path
model_path = r"C:\Users\reddy\Downloads\butterflies code\vgg16_model.tflite"

# ✅ Preprocess image
def preprocess_image_tflite(path):
    img = Image.open(path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img).astype(np.float32) / 255.0
    return img_array

# ✅ Load image
image_array = preprocess_image_tflite(file_path)
input_data = np.expand_dims(image_array, axis=0)

# ✅ Load and allocate TFLite model
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# ✅ Get input/output tensor info
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ✅ Set input and run inference
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# ✅ Get output
output_data = interpreter.get_tensor(output_details[0]['index'])
predicted_class = np.argmax(output_data)
butterfly_name = butterfly_names[predicted_class]

print(f"✅ Predicted Butterfly Class: {predicted_class}")
print(f"🦋 Butterfly Name: {butterfly_name}")
