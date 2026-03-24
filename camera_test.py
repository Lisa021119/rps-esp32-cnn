# =============================================================================
# camera_test.py
# Camera Initialization and Test Script - runs on ESP32S3 Sense
#
# Author: Lisa Chen
# Course: ENMGT 5400 - Project 4
#
# Description:
#   Verifies the OV2640 camera is properly connected and functional.
#   Captures a single test image and saves it to flash storage.
#   Transfer to laptop with Thonny to visually inspect image quality.
#
# Source: Based on Seeed Studio XIAO ESP32S3 Sense wiki example
# https://wiki.seeedstudio.com/XIAO_ESP32S3_Micropython/
# =============================================================================

from camera import Camera, GrabMode, PixelFormat, FrameSize, GainCeiling
import time

# =============================================================================
# CAMERA CONFIGURATION
# All pin numbers match Seeed Studio XIAO ESP32S3 Sense hardware schematic
# =============================================================================
CAMERA_PARAMETERS = {
    "data_pins": [15, 17, 18, 16, 14, 12, 11, 48],  # 8-bit parallel data bus
    "vsync_pin": 38,        # Frame sync signal
    "href_pin": 47,         # Line sync signal
    "sda_pin": 40,          # SCCB (I2C) data for camera register config
    "scl_pin": 39,          # SCCB (I2C) clock
    "pclk_pin": 13,         # Pixel clock output from camera
    "xclk_pin": 10,         # Master clock input to camera (must be provided)
    "xclk_freq": 20000000,  # 20 MHz - standard for OV2640
    "powerdown_pin": -1,    # Not used on this board
    "reset_pin": -1,        # Not used on this board
}

# =============================================================================
# INITIALIZE AND TEST
# =============================================================================
print("Initializing camera...")
cam = Camera(**CAMERA_PARAMETERS)
cam.init()
cam.set_bmp_out(True)   # BMP output: uncompressed, easier to process

# Print all camera settings for debugging
print("\nCamera properties:")
get_methods = [method for method in dir(cam) if callable(getattr(cam, method)) and method.startswith('get_')]
results = {}
for method in get_methods:
    try:
        value = getattr(cam, method)()
        results[method] = value
        print(f"  {method}: {value}")
    except Exception as e:
        print(f"  {method}: ERROR - {e}")

# Warm up: discard first few frames (camera needs time to auto-adjust exposure)
print("\nWarming up camera (3 frames)...")
for i in range(3):
    cam.capture()
    time.sleep(0.2)

# Capture test image
print("Capturing test image...")
buf = cam.capture()

if buf and len(buf) > 1000:
    # Save to flash
    with open('test_image.bmp', 'wb') as f:
        f.write(buf)
    print(f"SUCCESS: Captured image of size {len(buf)} bytes")
    print("Saved as test_image.bmp")
    print("Transfer to laptop: use Thonny Files panel -> right-click -> Download")
else:
    print("FAILED: Could not capture image")
    print("Check camera module is properly seated and lens cover removed")

cam.deinit()
print("\nCamera test complete.")
