# =============================================================================
# test_tmdl_from_camera.py
# Real-time Rock-Paper-Scissors Classification - runs on ESP32S3 Sense
#
# Author: Lisa Chen
# Course: ENMGT 5400 - Project 4
#
# Description:
#   Captures images from the OV2640 camera in real time, preprocesses them,
#   and runs inference using an emlearn CNN model (.tmdl format).
#   Prints classification result to serial (visible in Thonny shell on PC).
#
# Prerequisites:
#   1. Flash ESP32 with camera-enabled MicroPython firmware
#   2. Upload prs_cnn.tmdl to ESP32 root
#   3. Upload image_preprocessing.py to ESP32 root
#   4. Install emlearn_cnn_fp32: 
#      mpremote mip install https://emlearn.github.io/emlearn-micropython/builds/latest/xtensawin_6.3/emlearn_cnn_fp32.mpy
#
# Based on guidance from course instructor (Prof. Swart, ENMGT 5400)
# emlearn_cnn_fp32 library: https://github.com/emlearn/emlearn-micropython
# =============================================================================

import array
import gc
import time

# Camera module - available in camera-enabled MicroPython firmware
from camera import Camera, PixelFormat, FrameSize

# Image preprocessing helpers (provided by instructor)
# resize_96x96_to_32x32_and_threshold: downsamples 96x96 to 32x32 with thresholding
# strip_bmp_header: removes 54-byte BMP header to get raw pixel data
from image_preprocessing import resize_96x96_to_32x32_and_threshold
from image_preprocessing import strip_bmp_header

# emlearn CNN inference engine for MicroPython
# Runs fp32 CNN models stored in .tmdl format (TinyMaix format)
import emlearn_cnn_fp32 as emlearn_cnn

# =============================================================================
# CONFIGURATION
# =============================================================================
MODEL = 'prs_cnn.tmdl'          # Trained model file (must be on ESP32)
RECOGNITION_THRESHOLD = 0.74    # Minimum confidence to report a prediction
                                 # Below this = uncertain / not recognized

# Camera pin configuration for Seeed Studio XIAO ESP32S3 Sense
# These match the OV2640 camera module hardware connections
CAMERA_PARAMETERS = {
    "data_pins": [15, 17, 18, 16, 14, 12, 11, 48],
    "vsync_pin": 38,
    "href_pin": 47,
    "sda_pin": 40,
    "scl_pin": 39,
    "pclk_pin": 13,
    "xclk_pin": 10,
    "xclk_freq": 20000000,
    "powerdown_pin": -1,
    "reset_pin": -1,
    "frame_size": FrameSize.R96X96,         # Capture at 96x96 (smallest native)
    "pixel_format": PixelFormat.GRAYSCALE   # Grayscale: simpler, faster, enough for gestures
}

# Class labels - order must match training label order (alphabetical)
classes = ['paper', 'rock', 'scissors']

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def print_2d_buffer(buf, width, height):
    """Debug helper: prints pixel buffer as ASCII art to visualize image."""
    chars = ' .:-=+*#%@'
    for y in range(height):
        row = ''
        for x in range(width):
            pixel = buf[y * width + x]
            row += chars[pixel // 26]  # Map 0-255 to 10 chars
        print(row)

def argmax(arr):
    """
    Returns the index of the maximum value in an array.
    Used to find the most likely class from probability output.
    argmax([0.1, 0.8, 0.1]) -> 1 (index of 0.8)
    """
    max_val = arr[0]
    max_idx = 0
    for i in range(1, len(arr)):
        if arr[i] > max_val:
            max_val = arr[i]
            max_idx = i
    return max_idx

# =============================================================================
# INITIALIZATION
# =============================================================================

# Initialize camera
print("Initializing camera...")
cam = Camera(**CAMERA_PARAMETERS)
cam.init()
cam.set_bmp_out(True)   # Output BMP so we can strip header later
print("Camera ready")

# Load model from flash into RAM
# array.array('B') = unsigned byte array, efficient memory use on microcontroller
print(f"Loading model: {MODEL}")
with open(MODEL, 'rb') as f:
    model_data = array.array('B', f.read())
    print("Model data loaded:", len(model_data), "bytes")
    gc.collect()  # Free any fragmented memory before allocating model
    model = emlearn_cnn.new(model_data)
    print("Model initialized")

# Output buffer: one float per class
# emlearn_cnn.run() fills this with class probabilities
n_classes = len(classes)
probabilities = array.array('f', [0.0] * n_classes)

# Track current prediction to only print on change
current_prediction = classes[0]
cnt = 0  # Frame counter

print(f"\nStarting real-time classification...")
print(f"Classes: {classes}")
print(f"Confidence threshold: {RECOGNITION_THRESHOLD}")
print("=" * 40)

# =============================================================================
# MAIN INFERENCE LOOP
# =============================================================================
while True:
    cnt += 1

    # ----- Step 1: Capture image from camera -----
    raw_bmp = cam.capture()
    if not raw_bmp:
        print("Capture failed, retrying...")
        time.sleep(0.1)
        continue

    # ----- Step 2: Resize 96x96 -> 32x32 and apply threshold -----
    # Thresholding binarizes the image: pixels above threshold -> 255, below -> 0
    # This helps separate hand from background and reduces noise
    img_32x32 = resize_96x96_to_32x32_and_threshold(raw_bmp)

    # ----- Step 3: Strip BMP header -----
    # BMP files have a 54-byte header before raw pixel data
    # The CNN model expects raw pixel bytes only
    pixel_data = strip_bmp_header(img_32x32)

    # ----- Step 4: Run model inference -----
    # model.run() takes raw pixel bytes and fills probabilities buffer
    # Each value = confidence score for that class (0.0 to 1.0)
    model.run(pixel_data, probabilities)

    # ----- Step 5: Get prediction -----
    # argmax finds the class with highest probability
    best_idx = argmax(probabilities)
    best_confidence = probabilities[best_idx]
    best_class = classes[best_idx]

    # Only report if confidence exceeds threshold
    if best_confidence >= RECOGNITION_THRESHOLD:
        prediction = best_class
    else:
        prediction = "unknown"

    # Print result only when prediction changes (reduces serial spam)
    if prediction != current_prediction:
        current_prediction = prediction
        prob_str = ', '.join([f"{classes[i]}:{probabilities[i]:.2f}" for i in range(n_classes)])
        print(f"[{cnt:05d}] Prediction: {prediction.upper():10s} | {prob_str}")

    # Small delay to avoid overwhelming serial output
    time.sleep(0.1)
