import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

# Step 1: Data Preprocessing

train_cats_path = "dataset/train/cats"
train_dogs_path = "dataset/train/dogs"
test_cats_path = "dataset/test/cats"
test_dogs_path = "dataset/test/dogs"

IMG_HEIGHT = 64
IMG_WIDTH = 64
IMG_CHANNELS = 3

TRAIN_LIMIT = 1000
TEST_LIMIT = None

def load_images_from_folder(folder_path, label, limit=None):
    images = []
    labels = []

    file_names = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if limit is not None:
        file_names = file_names[:limit]

    for file_name in file_names:
        img_path = os.path.join(folder_path, file_name)

        img = tf.keras.utils.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
        img_array = tf.keras.utils.img_to_array(img)

        images.append(img_array)
        labels.append(label)

    return np.array(images), np.array(labels)

X_train_cats, y_train_cats = load_images_from_folder(train_cats_path, label=0, limit=TRAIN_LIMIT)
X_train_dogs, y_train_dogs = load_images_from_folder(train_dogs_path, label=1, limit=TRAIN_LIMIT)

X_train = np.concatenate((X_train_cats, X_train_dogs), axis=0)
y_train = np.concatenate((y_train_cats, y_train_dogs), axis=0)

X_test_cats, y_test_cats = load_images_from_folder(test_cats_path, label=0, limit=TEST_LIMIT)
X_test_dogs, y_test_dogs = load_images_from_folder(test_dogs_path, label=1, limit=TEST_LIMIT)

X_test = np.concatenate((X_test_cats, X_test_dogs), axis=0)
y_test = np.concatenate((y_test_cats, y_test_dogs), axis=0)

print("Training data shape:", X_train.shape)
print("Training labels shape:", y_train.shape)
print("Testing data shape:", X_test.shape)
print("Testing labels shape:", y_test.shape)

sample_img = X_train[0]
print("Min pixel value:", np.min(sample_img))
print("Max pixel value:", np.max(sample_img))

X_train = X_train / 255.0
X_test = X_test / 255.0

normalized_img = sample_img / 255.0

plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(sample_img.astype("uint8"))
plt.title("Before Normalization")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(normalized_img)
plt.title("After Normalization")
plt.axis("off")

plt.show()


# Step 2: Data Visualization

# Display sample images from the training set
plt.figure(figsize=(10, 6))

for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(X_train[i])
    
    if y_train[i] == 0:
        plt.title("Cat")
    else:
        plt.title("Dog")
    
    plt.axis("off")

plt.tight_layout()
plt.show()

# Show class distribution in training data
unique_classes, counts = np.unique(y_train, return_counts=True)

class_names = ["Cat", "Dog"]

plt.figure(figsize=(6, 4))
plt.bar(class_names, counts)
plt.title("Class Distribution in Training Data")
plt.xlabel("Classes")
plt.ylabel("Number of Images")
plt.show()

# Step 3: Model Design

# Build CNN model
model = Sequential([
    # First convolution layer
    # 32 filters, each 3x3, ReLU activation
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),

    # First pooling layer
    MaxPooling2D(2, 2),

    # Second convolution layer
    # 64 filters, each 3x3
    Conv2D(64, (3, 3), activation='relu'),

    # Second pooling layer
    MaxPooling2D(2, 2),

    # Flatten 2D feature maps into 1D vector
    Flatten(),

    # Fully connected hidden layer
    Dense(128, activation='relu'),

    # Output layer for binary classification
    # 1 neuron + sigmoid because this is cats vs dogs
    Dense(1, activation='sigmoid')
])

# Print model summary
model.summary()

# Step 4: Model Training

# Compile the CNN
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train the model using the training dataset
history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# Step 5: Evaluation

# Evaluate model on test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)

print("Test Accuracy:", test_accuracy)
print("Test Loss:", test_loss)

# Predict probabilities on test images
y_pred_probs = model.predict(X_test)

# Convert probabilities to class labels
# If probability >= 0.5, predict dog (1), else cat (0)
y_pred = (y_pred_probs > 0.5).astype("int32").flatten()

# Accuracy using sklearn
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=["Cat", "Dog"],
    yticklabels=["Cat", "Dog"]
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Plot training loss and validation loss
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Plot training accuracy and validation accuracy
plt.figure(figsize=(8, 4))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()