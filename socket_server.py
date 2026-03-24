# =============================================================================
# socket_server.py
# Image Streaming Server - runs on ESP32S3 Sense
#
# Author: Lisa Chen
# Course: ENMGT 5400 - Project 4
#
# Description:
#   Creates a TCP socket server on the ESP32 that continuously captures
#   camera frames and sends them to a connected laptop client.
#   Run socket_client.py on the laptop to receive and display the stream.
#
# Usage:
#   1. Update Wifi.py with your hotspot credentials
#   2. Run this script on ESP32 via Thonny
#   3. Note the IP address printed in Thonny shell
#   4. Run socket_client.py on your laptop with that IP
# =============================================================================

import socket
import time
from camera import Camera, PixelFormat, FrameSize
import Wifi

# =============================================================================
# CAMERA CONFIGURATION
# =============================================================================
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
    "frame_size": FrameSize.R96X96,
    "pixel_format": PixelFormat.GRAYSCALE
}

PORT = 5000     # TCP port to listen on
HEADER = 4      # Bytes used to send frame size before each frame

# =============================================================================
# CONNECT TO WIFI
# =============================================================================
ip = Wifi.connect()
if not ip:
    print("Cannot start server without WiFi connection")
    raise SystemExit

# =============================================================================
# INITIALIZE CAMERA
# =============================================================================
cam = Camera(**CAMERA_PARAMETERS)
cam.init()
cam.set_bmp_out(True)
print("Camera initialized")

# =============================================================================
# START SERVER
# =============================================================================
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(('0.0.0.0', PORT))
server.listen(1)

print(f"\nServer listening on {ip}:{PORT}")
print("Start socket_client.py on your laptop with this IP address")
print("WARNING: Board may get warm during streaming - this is normal")

# =============================================================================
# MAIN LOOP: Accept connections and stream frames
# =============================================================================
while True:
    print("\nWaiting for connection...")
    conn, addr = server.accept()
    print(f"Connected from: {addr}")
    
    frame_count = 0
    try:
        while True:
            # Capture frame
            buf = cam.capture()
            if not buf:
                continue
            
            # Send frame size as 4-byte big-endian integer
            # Client uses this to know how many bytes to read
            size = len(buf)
            conn.send(size.to_bytes(HEADER, 'big'))
            
            # Send frame data
            conn.send(buf)
            
            frame_count += 1
            if frame_count % 10 == 0:
                print(f"Frames sent: {frame_count}")
                
    except Exception as e:
        print(f"Connection closed: {e}")
        conn.close()
    
    print(f"Session ended. Total frames: {frame_count}")
