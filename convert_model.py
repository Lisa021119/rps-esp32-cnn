# =============================================================================
# convert_model.py
# Converts trained Keras model to TFLite format for ESP32 deployment
#
# Author: Lisa Chen
# Course: ENMGT 5400 - Project 4
#
# Description:
#   Takes the trained rps_model.h5 and converts it to:
#   1. rps_model.tflite  - standard TFLite (fp32)
#   2. rps_model_int8.tflite - quantized int8 (smaller, faster on ESP32)
#
# Usage:
#   python3 convert_model.py
# =============================================================================

import tensorflow as tf
import numpy as np
import os
import glob
from PIL import Image

# =============================================================================
# LOAD TRAINED MODEL
# =============================================================================
print("Loading trained model...")
model = tf.keras.models.load_model('rps_model.h5')
model.summary()

# =============================================================================
# CONVERT 1: Standard TFLite (fp32)
# Retains full float32 precision, larger file size
# =============================================================================
print("\nConverting to standard TFLite (fp32)...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_fp32 = converter.convert()

with open('rps_model.tflite', 'wb') as f:
    f.write(tflite_fp32)
print(f"Saved rps_model.tflite: {len(tflite_fp32)/1024:.1f} KB")

# =============================================================================
# CONVERT 2: Int8 Quantized TFLite
# Reduces model size by ~4x, faster inference on microcontrollers
# Requires representative dataset for calibration
# =============================================================================
print("\nConverting to int8 quantized TFLite...")

def representative_data_gen():
    """
    Provides sample images for quantization calibration.
    The converter uses these to determine optimal int8 scaling factors.
    """
    img_dir = os.path.expanduser('~/.keras/datasets/rps/rps')
    all_images = []
    for cls in ['rock', 'paper', 'scissors']:
        files = glob.glob(os.path.join(img_dir, cls, '*.png'))[:15]
        all_images.extend(files)

    for img_path in all_images:
        # Load, convert to grayscale, resize to 32x32
        img = Image.open(img_path).convert('L').resize((32, 32))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = img_array.reshape(1, 32, 32, 1)
        yield [img_array]

converter_int8 = tf.lite.TFLiteConverter.from_keras_model(model)
converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
converter_int8.representative_dataset = representative_data_gen
converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter_int8.inference_input_type = tf.uint8
converter_int8.inference_output_type = tf.uint8

tflite_int8 = converter_int8.convert()

with open('rps_model_int8.tflite', 'wb') as f:
    f.write(tflite_int8)
print(f"Saved rps_model_int8.tflite: {len(tflite_int8)/1024:.1f} KB")

# =============================================================================
# VERIFY BOTH MODELS
# =============================================================================
print("\nVerifying models...")

for model_path in ['rps_model.tflite', 'rps_model_int8.tflite']:
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()
    out = interpreter.get_output_details()
    print(f"\n{model_path}:")
    print(f"  Input shape:  {inp[0]['shape']} dtype={inp[0]['dtype']}")
    print(f"  Output shape: {out[0]['shape']} dtype={out[0]['dtype']}")

print("\nConversion complete!")
print("Use rps_model.tflite or rps_model_int8.tflite for ESP32 deployment.")
