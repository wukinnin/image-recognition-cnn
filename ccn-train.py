## Imports
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import keras
import numpy as np
import cv2
import tensorflow as tf

from random import shuffle
from keras import Input

Sequential = tf.keras.models.Sequential
Conv2D = tf.keras.layers.Conv2D
MaxPooling2D = tf.keras.layers.MaxPooling2D
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
Flatten = tf.keras.layers.Flatten

# Optimizer
Adam = tf.keras.optimizers.Adam

## Directories and hyperparameters
TRAIN_DIR = 'train'
IMG_SIZE = 128
IMAGE_CHANNELS = 3
FIRST_NUM_CHANNEL = 32
FILTER_SIZE = 3
LR = 0.0001
PERCENT_TRAINING_DATA = 80
NUM_EPOCHS = 50
MODEL_NAME = 'animals_cnn'

# Function to read all classes in the train folder
def define_classes():
    all_classes = []
    for folder in os.listdir(TRAIN_DIR):
        all_classes.append(folder)
    return all_classes, len(all_classes)

# Function to define labels as one-hot encoded arrays
def define_labels(all_classes):
    all_labels = []
    for x in range(len(all_classes)):
        label = np.zeros(len(all_classes))
        label[x] = 1
        all_labels.append(label)
    return all_labels

# Function to load and preprocess images
def create_train_data(all_classes, all_labels):
    training_data = []
    for label_index, specific_class in enumerate(all_classes):
        current_dir = os.path.join(TRAIN_DIR, specific_class)
        print(f'Reading directory: {current_dir}')
        for img_filename in os.listdir(current_dir):
            path = os.path.join(current_dir, img_filename)
            
            # Ensure only valid image files are processed
            if not img_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                print(f"Skipping non-image file: {path}")
                continue
            
            img = cv2.imread(path)
            
            # Handle failed image loading
            if img is None:
                print(f"Warning: Unable to read image at {path}")
                continue
            
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            training_data.append([img, all_labels[label_index]])
    shuffle(training_data)
    return training_data

# Prepare data
all_classes, NUM_OUTPUT = define_classes()
all_labels = define_labels(all_classes)
training_data = create_train_data(all_classes, all_labels)

# Split into training and test sets
train = training_data[:int(len(training_data) * (PERCENT_TRAINING_DATA / 100))]
test = training_data[-int(len(training_data) * (PERCENT_TRAINING_DATA / 100)):]
X_train = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, IMAGE_CHANNELS) / 255.0
Y_train = np.array([i[1] for i in train])
X_test = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, IMAGE_CHANNELS) / 255.0
Y_test = np.array([i[1] for i in test])

# Build the model
model = Sequential([
    Input(shape=(IMG_SIZE, IMG_SIZE, IMAGE_CHANNELS)),  # Explicit Input layer
    Conv2D(FIRST_NUM_CHANNEL, (FILTER_SIZE, FILTER_SIZE), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(FIRST_NUM_CHANNEL * 2, (FILTER_SIZE, FILTER_SIZE), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(FIRST_NUM_CHANNEL * 4, (FILTER_SIZE, FILTER_SIZE), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(FIRST_NUM_CHANNEL * 8, (FILTER_SIZE, FILTER_SIZE), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(FIRST_NUM_CHANNEL * 16, activation='relu'),
    Dropout(0.8),
    Dense(NUM_OUTPUT, activation='softmax')
])


# Compile the model
model.compile(optimizer=Adam(learning_rate=LR), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, Y_train, epochs=NUM_EPOCHS, batch_size=32, validation_data=(X_test, Y_test), verbose=1)

# Save the model
model.save(f"{MODEL_NAME}.keras")

