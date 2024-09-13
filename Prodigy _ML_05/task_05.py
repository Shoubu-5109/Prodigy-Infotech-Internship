import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# Path to your food dataset
dataset_path = 'path_to_food_dataset'

# Preprocess the images and labels
image_size = (128, 128)

def load_food_images(folder):
    images = []
    labels = []
    class_names = os.listdir(folder)
    for class_index, class_name in enumerate(class_names):
        class_folder = os.path.join(folder, class_name)
        for filename in os.listdir(class_folder):
            img_path = os.path.join(class_folder, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, image_size)
                images.append(img)
                labels.append(class_index)
    return np.array(images), np.array(labels), class_names

images, labels, class_names = load_food_images(dataset_path)

# Normalize the images
images = images / 255.0

# Split into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils
model = models.Sequential()

# Add Convolutional layers
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Add Flatten and Dense layers
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(len(class_names), activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
history = model.fit(x_train, y_train, epochs=15, validation_data=(x_test, y_test), batch_size=32)
