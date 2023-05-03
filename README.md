# mnist-neural-network
  This is a Python script that trains a convolutional neural network to recognize handwritten digits from the MNIST dataset. It uses the Keras deep learning library with a TensorFlow backend.
    The neural network has two convolutional layers, followed by two fully connected layers. It uses the ReLU activation function for the hidden layers and the softmax activation function for the output layer. The loss function is sparse categorical crossentropy, and the optimizer is Adam.

## Usage
    Clone the repository to your local machine.
        git clone https://github.com/ayraqutub/mnist-neural-network.git
    Install the necessary packages.
        pip install -r requirements.txt
    Run the Python script.
        python mnist.py
    The script will train the neural network on the MNIST dataset and display the training and validation loss over epochs, as well as a set of 20 random images from the training set with their predicted labels.
    Note: The script does not evaluate the model on the test set. To do so, modify the code to include an evaluation step using the evaluate method.

## Acknowledgments
    The MNIST dataset is a collection of handwritten digits that has been widely used for image classification tasks. It was created by Yann LeCun, Corinna Cortes, and Christopher J.C. Burges for the National Institute of Standards and Technology (NIST) in the 1990s.
    This code was created through the Coursera Machine Learning course as taught by Jouseph Murad
