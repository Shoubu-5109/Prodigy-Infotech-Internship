import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
# Set the path for the dataset
cat_folder = 'path_to_cats_folder'
dog_folder = 'path_to_dogs_folder'

IMG_SIZE = 64  # Image size (64x64 pixels)

def load_images(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        if filename.endswith(".jpg"):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize image to 64x64
            images.append(img)
            label = 1 if 'dog' in folder else 0  # Label: 1 for dog, 0 for cat
            labels.append(label)
    return np.array(images), np.array(labels)

# Load images
cat_images, cat_labels = load_images(cat_folder)
dog_images, dog_labels = load_images(dog_folder)

# Combine data
X = np.concatenate((cat_images, dog_images), axis=0)
y = np.concatenate((cat_labels, dog_labels), axis=0)

# Normalize pixel values (0 to 1)
X = X / 255.0

# Reshape data to 1D array per image
X = X.reshape(X.shape[0], -1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Create an SVM model with a linear kernel
svm_model = SVC(kernel='linear')

# Train the model
svm_model.fit(X_train, y_train)
# Predict the labels for the test set
y_pred = svm_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
def display_predictions(X_test, y_test, y_pred, num_samples=5):
    for i in range(num_samples):
        plt.imshow(X_test[i].reshape(IMG_SIZE, IMG_SIZE), cmap='gray')
        plt.title(f"True: {'Dog' if y_test[i] == 1 else 'Cat'}, Pred: {'Dog' if y_pred[i] == 1 else 'Cat'}")
        plt.show()

display_predictions(X_test, y_test, y_pred)
