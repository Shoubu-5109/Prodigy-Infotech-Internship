import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# Path to your dataset
dataset_path = 'path_to_dataset'

# Preprocess the images and labels
image_size = (64, 64)  # Resize images to 64x64

def load_images_from_folder(folder):
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
    return np.array(images), np.array(labels)

images, labels = load_images_from_folder(dataset_path)

# Normalize pixel values to [0, 1]
images = images / 255.0

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)
model = models.Sequential()

# Add Convolutional layers
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Add Flatten and Dense layers
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))  # Add dropout to avoid overfitting
model.add(layers.Dense(len(os.listdir(dataset_path)), activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
history = model.fit(x_train, y_train, epochs=15, validation_data=(x_test, y_test), batch_size=32)
# Evaluate on test data
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc}')

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()
# Load the trained model
model = tf.keras.models.load_model('path_to_saved_model')

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    img = cv2.resize(frame, image_size)
    img = np.expand_dims(img, axis=0)
    
    # Predict gesture
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)

    # Show the predicted gesture
    cv2.putText(frame, f'Gesture: {predicted_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    # Display the video feed
    cv2.imshow('Hand Gesture Recognition', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
