import numpy as np
import matplotlib.pyplot as plt

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
