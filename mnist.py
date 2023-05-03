import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load the MNIST dataset
(x_train_full, y_train_full), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize the data
x_train_full = x_train_full.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# Reshape the data to 4D arrays for use with convolutions
x_train_full = x_train_full.reshape((len(x_train_full), 28, 28, 1))
x_test = x_test.reshape((len(x_test), 28, 28, 1))

# Convert the labels to one-hot encoded vectors
y_train_full = keras.utils.to_categorical(y_train_full)
y_test = keras.utils.to_categorical(y_test)

# Split the training set into training and validation sets
split_index = int(0.8 * len(x_train_full))
x_train, x_val = x_train_full[:split_index], x_train_full[split_index:]
y_train, y_val = y_train_full[:split_index], y_train_full[split_index:]

# Define the model architecture
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model with categorical cross-entropy loss function and Adam optimizer
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model on the training data with a validation set
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_val, y_val), verbose=2)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc:.3f}')


# Train the model on the training data with a validation set
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_val, y_val), verbose=2)

# Plot the training and validation loss over epochs
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Get predicted labels for 20 random images from the training set
random_indices = np.random.choice(len(x_train), size=20, replace=False)
x_random = x_train[random_indices]
y_random = y_train[random_indices]
y_random_pred = model.predict(x_random)
y_random_pred_classes = np.argmax(y_random_pred, axis=1)

# Plot the random images with predicted labels
fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(10, 8))
for i, ax in enumerate(axes.flat):
    ax.imshow(x_random[i].reshape(28, 28), cmap='binary')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Pred: {}'.format(y_random_pred_classes[i]))
    if y_random_pred_classes[i] == np.argmax(y_random[i]):
        ax.spines['bottom'].set_color('green')
        ax.spines['top'].set_color('green')
        ax.spines['right'].set_color('green')
        ax.spines['left'].set_color('green')
    else:
        ax.spines['bottom'].set_color('red')
        ax.spines['top'].set_color('red')
        ax.spines['right'].set_color('red')
        ax.spines['left'].set_color('red')
plt.tight_layout()
plt.show()
