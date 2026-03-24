# Rock-Paper-Scissors CNN Classifier
**ENMGT 5400 - Project 4 | Lisa Chen | Spring 2026**

A real-time hand gesture recognition system using a Convolutional Neural Network deployed on the Seeed Studio XIAO ESP32S3 Sense microcontroller.

## Project Overview

This project trains a CNN to classify rock, paper, and scissors hand gestures captured by the ESP32's OV2640 camera, and runs inference entirely on-device in MicroPython.

## Repository Structure

```
├── train_cnn.py              # Train CNN on laptop (runs on Mac/PC)
├── convert_model.py          # Convert .h5 model to TFLite/.tmdl for ESP32
├── collect_data.py           # Capture training images from ESP32 camera
├── test_tmdl_from_camera.py  # Real-time classification on ESP32
├── socket_server.py          # ESP32: stream camera frames over WiFi
├── socket_client.py          # Laptop: receive and display camera stream
├── camera_test.py            # ESP32: test camera initialization
├── Wifi.py                   # ESP32: WiFi connection helper
├── image_preprocessing.py    # ESP32: resize, threshold, strip BMP header
└── RPS_Design_Documentation.docx  # Full design document
```

## CNN Architecture

| Layer | Type | Output Shape | Parameters |
|-------|------|-------------|------------|
| Input | Input | 32x32x1 | 0 |
| conv1 | Conv2D (16, 3x3, ReLU) | 32x32x16 | 160 |
| pool1 | MaxPooling2D (2x2) | 16x16x16 | 0 |
| conv2 | Conv2D (32, 3x3, ReLU) | 16x16x32 | 4,640 |
| pool2 | MaxPooling2D (2x2) | 8x8x32 | 0 |
| conv3 | Conv2D (64, 3x3, ReLU) | 8x8x64 | 18,496 |
| pool3 | MaxPooling2D (2x2) | 4x4x64 | 0 |
| flatten | Flatten | 1024 | 0 |
| dropout1 | Dropout (0.3) | 1024 | 0 |
| dense1 | Dense (128, ReLU) | 128 | 131,200 |
| dropout2 | Dropout (0.3) | 128 | 0 |
| output | Dense (3, Softmax) | 3 | 387 |

**Total parameters: 154,883**  
**Validation accuracy: >95%**

## Hardware

- **Board**: Seeed Studio XIAO ESP32S3 Sense
- **Camera**: OV2640 (96x96 minimum, downsampled to 32x32)
- **Firmware**: Camera-enabled MicroPython (shariltumin build)

## Setup Instructions

### 1. Flash ESP32 Firmware
```bash
# Use Thonny: Run -> Configure Interpreter -> Install MicroPython
# Select local firmware file from Canvas (camera-enabled build)
# Put ESP32 in bootloader mode: hold BOOT + press RESET
```

### 2. Train CNN on Laptop
```bash
pip3 install tensorflow numpy pillow matplotlib
python3 train_cnn.py
```

### 3. Convert Model
```bash
python3 convert_model.py
# Outputs: rps_model.tflite, rps_model_int8.tflite
```

### 4. Collect Training Data (optional - for custom images)
- Upload `collect_data.py` to ESP32 via Thonny
- Change `CLASS_NAME` to `'rock'`, `'paper'`, or `'scissors'`
- Run and hold hand gesture in front of camera

### 5. Deploy to ESP32
```bash
# Install emlearn CNN module
mpremote connect /dev/tty.usbmodem101 mip install \
  https://emlearn.github.io/emlearn-micropython/builds/latest/xtensawin_6.3/emlearn_cnn_fp32.mpy

# Upload model and inference script via Thonny Files panel
```

### 6. Run Real-time Classification
- Upload `test_tmdl_from_camera.py` to ESP32
- Run via Thonny - results print in Shell window

## Dataset

- **TensorFlow public RPS dataset**: 2,520 images (840 per class)
- **Custom-captured images**: 201 images from project's OV2640 camera
- **Image format**: 32x32 grayscale BMP

## Dependencies

### Laptop
- Python 3.x, TensorFlow, NumPy, Pillow, matplotlib, opencv-python

### ESP32 (MicroPython)
- Camera-enabled MicroPython firmware
- `emlearn_cnn_fp32.mpy` (emlearn-micropython library)
- `image_preprocessing.py` (provided by instructor)

## References

- Seeed Studio XIAO ESP32S3 MicroPython Wiki
- emlearn-micropython: https://github.com/emlearn/emlearn-micropython
- Camera firmware (shariltumin): https://github.com/shariltumin/esp32-cam-micropython-2022
- TensorFlow RPS dataset: https://storage.googleapis.com/download.tensorflow.org/data/rps.zip
- `image_preprocessing.py` provided by Prof. Swart, ENMGT 5400
