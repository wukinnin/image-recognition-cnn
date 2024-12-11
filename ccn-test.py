import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
import keras
import tensorflow

from tensorflow.keras.models import load_model

# Parameters
IMG_SIZE = 128  # Matches the size used in training
IMAGE_CHANNELS = 3
MODEL_NAME = 'animals_cnn.keras'  # Saved model file name
TEST_IMG = 'cat146-test.jpg'  # Test image file
TRAIN_DIR = 'train'  # Directory containing subdirectories for each class

# Function to define classes dynamically from training directory
def define_classes(train_dir):
    all_classes = []
    for folder in os.listdir(train_dir):
        if os.path.isdir(os.path.join(train_dir, folder)):
            all_classes.append(folder)
    return sorted(all_classes)  # Ensure consistent class order

# Load the trained model
if not os.path.exists(MODEL_NAME):
    raise FileNotFoundError(f"Model file '{MODEL_NAME}' not found. Check the file path.")
print(f"Loading model: {MODEL_NAME}")
model = load_model(MODEL_NAME)

# Get the class labels
all_classes = define_classes(TRAIN_DIR)
if not all_classes:
    raise ValueError(f"No classes found in directory '{TRAIN_DIR}'. Ensure it contains subdirectories for each class.")
print(f"Classes: {all_classes}")

# Preprocess the test image
if not os.path.exists(TEST_IMG):
    raise FileNotFoundError(f"Test image '{TEST_IMG}' not found. Check the file path.")
print(f"Loading test image: {TEST_IMG}")
test_img = cv2.imread(TEST_IMG)
test_img = cv2.resize(test_img, (IMG_SIZE, IMG_SIZE))  # Resize to match model input
test_img = test_img.reshape(1, IMG_SIZE, IMG_SIZE, IMAGE_CHANNELS) / 255.0  # Normalize pixel values

# Perform prediction
predictions = model.predict(test_img)
predicted_class_index = np.argmax(predictions[0])  # Index of the highest probability
predicted_class = all_classes[predicted_class_index]
confidence = predictions[0][predicted_class_index] * 100

# Display the results
print(f"Predicted Class: {predicted_class}")
print(f"Confidence: {confidence:.2f}%")
print("\nClass Probabilities:")
for i, class_name in enumerate(all_classes):
    print(f"{class_name}: {predictions[0][i] * 100:.2f}%")
