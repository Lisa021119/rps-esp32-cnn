# =============================================================================
# collect_data.py
# Image Collection Script - runs on ESP32S3 Sense via MicroPython
#
# Author: Lisa Chen
# Course: ENMGT 5400 - Project 4
#
# Description:
#   Captures images from the OV2640 camera and saves them to the ESP32
#   internal flash for use as a training dataset. Images are saved as
#   BMP files (uncompressed) to avoid JPEG decoding complexity.
#
# Usage:
#   1. Change CLASS_NAME to 'rock', 'paper', or 'scissors'
#   2. Upload to ESP32 using Thonny
#   3. Run - hold hand gesture in front of camera
#   4. Transfer images to laptop using mpremote
#
# Camera pins follow Seeed Studio XIAO ESP32S3 Sense pinout
# =============================================================================

from camera import Camera, PixelFormat, FrameSize
import time
import os

# =============================================================================
# CONFIGURATION - change this before each recording session
# =============================================================================
CLASS_NAME = 'rock'     # Change to: 'rock', 'paper', or 'scissors'
TARGET_COUNT = 100      # Number of images to capture per class
CAPTURE_INTERVAL = 2    # Seconds between each capture (time to adjust pose)

# =============================================================================
# CAMERA CONFIGURATION
# Pin assignments for Seeed Studio XIAO ESP32S3 Sense
# These match the OV2640 camera module wiring on this specific board
# =============================================================================
CAMERA_PARAMETERS = {
    "data_pins": [15, 17, 18, 16, 14, 12, 11, 48],  # D0-D7 data bus
    "vsync_pin": 38,        # Vertical sync
    "href_pin": 47,         # Horizontal reference
    "sda_pin": 40,          # I2C data (camera config)
    "scl_pin": 39,          # I2C clock (camera config)
    "pclk_pin": 13,         # Pixel clock
    "xclk_pin": 10,         # External clock input to camera
    "xclk_freq": 20000000,  # 20MHz clock frequency
    "powerdown_pin": -1,    # Not connected on this board
    "reset_pin": -1,        # Not connected on this board
}

# =============================================================================
# SETUP
# =============================================================================
# Create output directory for this class
try:
    os.mkdir(CLASS_NAME)
    print("Created folder:", CLASS_NAME)
except OSError:
    print("Folder already exists:", CLASS_NAME)

# Check available storage space
st = os.statvfs('/')
free_kb = st[0] * st[3] / 1024
print(f"Free flash storage: {free_kb:.0f} KB")

if free_kb < 500:
    print("WARNING: Low storage space! Transfer existing files first.")

# Initialize camera
cam = Camera(**CAMERA_PARAMETERS)
cam.init()
cam.set_bmp_out(True)   # Output BMP format (includes header, uncompressed)
print("Camera initialized successfully")

# =============================================================================
# CAPTURE LOOP
# =============================================================================
print(f"\nCapturing {TARGET_COUNT} images for class: [{CLASS_NAME}]")
print(f"Interval: {CAPTURE_INTERVAL}s per image")
print("Starting in 3 seconds - get your hand ready!")
time.sleep(3)

count = 0
errors = 0

while count < TARGET_COUNT:
    # Capture frame from camera
    buf = cam.capture()

    if buf and len(buf) > 1000:  # Sanity check: valid image should be >1KB
        # Save as BMP file with zero-padded index for sorting
        filename = f"{CLASS_NAME}/{CLASS_NAME}_{count:03d}.bmp"
        with open(filename, 'wb') as f:
            f.write(buf)
        count += 1
        print(f"  [{count:3d}/{TARGET_COUNT}] Saved: {filename} ({len(buf)} bytes)")
    else:
        errors += 1
        print(f"  Capture failed (attempt {errors}), retrying...")
        if errors > 10:
            print("Too many errors - check camera connection")
            break

    # Wait before next capture (gives time to slightly adjust hand position
    # to create variety in the training data)
    time.sleep(CAPTURE_INTERVAL)

# =============================================================================
# DONE
# =============================================================================
print(f"\nCapture complete: {count} images saved to /{CLASS_NAME}/")
print(f"Transfer to laptop: mpremote connect /dev/tty.usbmodem101 cp -r :{CLASS_NAME}/ ~/data/{CLASS_NAME}/")
cam.deinit()
