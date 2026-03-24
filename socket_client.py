# =============================================================================
# socket_client.py
# Image Stream Receiver - runs on laptop/Mac
#
# Author: Lisa Chen
# Course: ENMGT 5400 - Project 4
#
# Description:
#   Connects to the ESP32 socket server and displays the live camera stream.
#   Also saves frames to disk for building the training dataset.
#
# Requirements:
#   pip3 install opencv-python numpy
#
# Usage:
#   python3 socket_client.py --ip 192.168.x.x
# =============================================================================

import socket
import numpy as np
import cv2
import os
import argparse
import time

# =============================================================================
# CONFIGURATION
# =============================================================================
PORT = 5000
HEADER = 4          # Bytes for frame size prefix
IMG_SIZE = 96       # Camera output size (96x96 grayscale)
SAVE_FRAMES = False # Set True to save frames as dataset
SAVE_DIR = "captured_frames"

def recv_all(sock, n):
    """
    Receive exactly n bytes from socket.
    Handles partial reads which are common with TCP.
    """
    data = bytearray()
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError("Connection closed by server")
        data.extend(chunk)
    return bytes(data)

def main(esp_ip):
    """Main loop: receive frames from ESP32 and display."""
    
    if SAVE_FRAMES:
        os.makedirs(SAVE_DIR, exist_ok=True)
        print(f"Saving frames to: {SAVE_DIR}/")
    
    print(f"Connecting to ESP32 at {esp_ip}:{PORT}...")
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((esp_ip, PORT))
    print("Connected! Press 'q' to quit, 's' to save current frame")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            # Read frame size (4 bytes)
            size_bytes = recv_all(sock, HEADER)
            frame_size = int.from_bytes(size_bytes, 'big')
            
            # Read frame data
            frame_data = recv_all(sock, frame_size)
            
            # BMP header is 54 bytes for grayscale BMP
            # Skip header to get raw pixel data
            BMP_HEADER_SIZE = 54
            if len(frame_data) > BMP_HEADER_SIZE:
                pixels = np.frombuffer(frame_data[BMP_HEADER_SIZE:], dtype=np.uint8)
                
                # Reshape to 96x96 image
                # Note: BMP stores rows bottom-to-top, flip vertically
                if len(pixels) >= IMG_SIZE * IMG_SIZE:
                    img = pixels[:IMG_SIZE * IMG_SIZE].reshape(IMG_SIZE, IMG_SIZE)
                    img = np.flipud(img)  # Correct BMP vertical flip
                    
                    # Display (enlarge for visibility)
                    display = cv2.resize(img, (384, 384), interpolation=cv2.INTER_NEAREST)
                    cv2.imshow('ESP32 Camera Stream', display)
                    
                    # Save frame if enabled
                    if SAVE_FRAMES:
                        cv2.imwrite(f"{SAVE_DIR}/frame_{frame_count:05d}.png", img)
                    
                    frame_count += 1
                    
                    # Show FPS every 30 frames
                    if frame_count % 30 == 0:
                        elapsed = time.time() - start_time
                        fps = frame_count / elapsed
                        print(f"Frames: {frame_count} | FPS: {fps:.1f}")
            
            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite(f"saved_{frame_count:05d}.png", img)
                print(f"Saved frame {frame_count}")
                
    except Exception as e:
        print(f"Error: {e}")
    finally:
        sock.close()
        cv2.destroyAllWindows()
        print(f"\nTotal frames received: {frame_count}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', required=True, help='ESP32 IP address (shown in Thonny)')
    args = parser.parse_args()
    main(args.ip)
