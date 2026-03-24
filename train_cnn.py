# =============================================================================
# train_cnn.py
# Rock-Paper-Scissors CNN Classifier - Training Script (runs on laptop/Mac)
#
# Author: Lisa Chen
# Course: ENMGT 5400 - Project 4
#
# Description:
#   Trains a Convolutional Neural Network to classify rock, paper, scissors
#   hand gestures. Uses 32x32 grayscale images to match ESP32S3 camera output.
#   Saves model as both .h5 (Keras) and .tflite (for ESP32 deployment).
#
# Usage:
#   python3 train_cnn.py
# =============================================================================

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os

print("TensorFlow version:", tf.__version__)

# =============================================================================
# CONFIGURATION
# =============================================================================
IMG_SIZE = 32       # 32x32 pixels - matches ESP32 downsampled output
BATCH_SIZE = 32     # Number of images per training batch
EPOCHS = 20         # Number of full passes through the training data
DATA_DIR = os.path.expanduser("~/.keras/datasets/rps/rps")  # Public dataset

# =============================================================================
# STEP 1: LOAD AND PREPROCESS DATA
# =============================================================================
print("\nLoading dataset from:", DATA_DIR)
print("Image size:", IMG_SIZE, "x", IMG_SIZE, "(grayscale)")

# Training set: 80% of data
train_dataset = keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,       # Reserve 20% for validation
    subset="training",
    seed=42,                    # Fixed seed for reproducibility
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    color_mode="grayscale"      # Grayscale matches ESP32 camera output
)

# Validation set: 20% of data (used to evaluate generalization)
val_dataset = keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    color_mode="grayscale"
)

# Get class names (rock, paper, scissors - alphabetical order)
class_names = train_dataset.class_names
print("Classes found:", class_names)

# Normalize pixel values from [0, 255] to [0, 1]
# This helps the neural network converge faster during training
normalization_layer = layers.Rescaling(1./255)
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y))

# Performance optimization: cache data in memory and prefetch next batch
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# =============================================================================
# STEP 2: BUILD THE CNN MODEL
# =============================================================================
# Architecture: 3 convolutional blocks + 2 fully connected layers
# Each conv block: Conv2D (feature extraction) -> MaxPooling (downsampling)
# This progressively extracts higher-level features while reducing spatial size

print("\nBuilding CNN model...")

model = keras.Sequential([

    # --- Input ---
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),  # 32x32x1 grayscale image

    # --- Convolutional Block 1 ---
    # Conv2D: 16 filters, 3x3 kernel, ReLU activation
    # Learns low-level features: edges, corners, basic shapes
    # Output: 32x32x16
    layers.Conv2D(16, (3, 3), activation='relu', padding='same', name='conv1'),
    # MaxPooling: reduces spatial dimensions by 2x
    # Output: 16x16x16
    layers.MaxPooling2D(2, 2, name='pool1'),

    # --- Convolutional Block 2 ---
    # Conv2D: 32 filters - learns more complex features (finger shapes, outlines)
    # Output: 16x16x32
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv2'),
    # Output: 8x8x32
    layers.MaxPooling2D(2, 2, name='pool2'),

    # --- Convolutional Block 3 ---
    # Conv2D: 64 filters - learns high-level features (hand posture patterns)
    # Output: 8x8x64
    layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv3'),
    # Output: 4x4x64
    layers.MaxPooling2D(2, 2, name='pool3'),

    # --- Flatten ---
    # Converts 4x4x64 = 1024 feature map into a 1D vector
    layers.Flatten(name='flatten'),

    # --- Regularization ---
    # Dropout: randomly disables 30% of neurons during training
    # Prevents overfitting (memorizing training data instead of generalizing)
    layers.Dropout(0.3, name='dropout1'),

    # --- Fully Connected Layer ---
    # Dense: 128 neurons, combines learned features for classification
    layers.Dense(128, activation='relu', name='dense1'),
    layers.Dropout(0.3, name='dropout2'),

    # --- Output Layer ---
    # 3 neurons (one per class), softmax converts to probabilities summing to 1
    layers.Dense(3, activation='softmax', name='output')
])

# Print model architecture summary
model.summary()

# =============================================================================
# STEP 3: COMPILE AND TRAIN
# =============================================================================
# Adam optimizer: adaptive learning rate, works well for most problems
# sparse_categorical_crossentropy: loss function for multi-class classification
# accuracy: metric we care about for this project

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(f"\nStarting training for {EPOCHS} epochs...")
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    verbose=1
)

# =============================================================================
# STEP 4: EVALUATE AND SAVE
# =============================================================================
val_loss, val_acc = model.evaluate(val_dataset, verbose=0)
print(f"\nFinal validation accuracy: {val_acc*100:.1f}%")

if val_acc > 0.75:
    print("EXCELLENT - Above 75% bonus threshold!")
elif val_acc > 0.5:
    print("PASS - Above 50% requirement")
else:
    print("FAIL - Below 50%, consider more epochs or data augmentation")

# Save Keras model (for reloading and further training)
model.save('rps_model.h5')
print("\nModel saved as rps_model.h5")

# Save class names for reference
with open('class_names.txt', 'w') as f:
    for name in class_names:
        f.write(name + '\n')
print("Class names saved:", class_names)

# =============================================================================
# STEP 5: CONVERT TO TFLITE (for ESP32 deployment)
# =============================================================================
print("\nConverting to TFLite format for ESP32 deployment...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
# DEFAULT optimization: quantizes weights, reduces size ~4x with minimal accuracy loss
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('rps_model.tflite', 'wb') as f:
    f.write(tflite_model)
print(f"TFLite model saved: {len(tflite_model)/1024:.1f} KB")

# =============================================================================
# STEP 6: PLOT TRAINING CURVES
# =============================================================================
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150)
print("Training curves saved as training_curves.png")
