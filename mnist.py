import numpy as np
from neural_network import *
import util
import random

# Initialise a new neural network
LAYER_SIZES = [28*28,800,10]
BATCH_SIZE = 100
neural_network = NeuralNetwork(LAYER_SIZES, BATCH_SIZE, LeakyReLU(0.0), SoftmaxCrossEntropy())

# Load a neural network from memory
#  neural_network = util.load("mnist.nn")

# Train the neural network
training_images = util.read_MNIST_image_file_ex("./data/train-images-idx3-ubyte")
training_labels = util.read_MNIST_label_file("./data/train-labels-idx1-ubyte")
training_data = list(zip(training_images, training_labels))

neural_network.train(training_data)

# Save a neural network to memory
util.save(neural_network, "mnist.nn")

# Test the neural network
test_images = util.read_MNIST_image_file_ex("./data/t10k-images-idx3-ubyte")
test_labels = util.read_MNIST_label_file("./data/t10k-labels-idx1-ubyte")
test_data = list(zip(test_images, test_labels))

neural_network.evaluate(test_data)
