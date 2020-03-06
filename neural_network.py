import numpy as np
import math
import random
from time import time
import util


# np.random.seed(0)

class ActivationFunction:
    @staticmethod
    def activation_function(x):
        raise NotImplementedError
    
    @staticmethod
    def derivative(x):
        raise NotImplementedError
        
    def __call__(self, x):
        return self.activation_function(x)

class Sigmoid(ActivationFunction):
    @staticmethod
    def activation_function(x):
        return 1.0 / (1.0 + np.exp(-x))
    
    @staticmethod
    def derivative(x):
        temp = 1.0 / (1.0 + np.exp(-x))
        return temp * (1.0 - temp)

class LeakyReLU(ActivationFunction):
    def __init__(self, leakage_coefficient=0.01):
        self.leakage_coefficient = leakage_coefficient

    def activation_function(self, x):
        return np.where(x > 0, x, self.leakage_coefficient*x)
    
    def derivative(self, x):
        return np.where(x > 0, 1.0, self.leakage_coefficient)

class FinalLayerHandler:
    @staticmethod
    def activation(z):
        raise NotImplementedError
    
    @staticmethod
    def error(a, t):
        raise NotImplementedError
    
    def total_error(self, a, t):
        return np.sum(self.error(a, t))
    
    @staticmethod
    def combined_derivative(z, a, t):
        raise NotImplementedError

class SoftmaxCrossEntropy(FinalLayerHandler):
    @staticmethod
    def activation(z):
        temp = np.exp(z)
        return temp / np.sum(temp)
    
    @staticmethod
    def error(a, t):
        return -t*np.log(a)
    
    @staticmethod
    def combined_derivative(z, a, t):
        return a - t

class NeuralNetwork():
    def __init__(self, layer_sizes, batch_size, activation, final_layer_handler, learning_rate=1.0):
        self.layer_sizes = layer_sizes
        self.batch_size = batch_size
        self.n_layers = len(layer_sizes)
        self.activation = activation
        self.final_layer_handler = final_layer_handler
        self.learning_rate = learning_rate

        self.ws = []
        self.bs = []
        for i in range(self.n_layers-1):
            self.ws.append(np.random.standard_normal((layer_sizes[i+1], layer_sizes[i])) * math.sqrt(2.0 / layer_sizes[i]))
            self.bs.append(np.zeros((layer_sizes[i+1], 1)))

    def forward_pass(self, input_layer):
        raw_layers = [None]
        activated_layers = [input_layer]
        for i in range(self.n_layers-2):
            raw_layers.append(self.ws[i] @ activated_layers[i] + self.bs[i])
            activated_layers.append(self.activation(raw_layers[i+1]))

        raw_layers.append(self.ws[-1] @ activated_layers[-1] + self.bs[-1])
        activated_layers.append(self.final_layer_handler.activation(raw_layers[-1]))

        return raw_layers, activated_layers

    # Just returns the output
    def run(self, input_layer):
        _, activated_layers = self.forward_pass(input_layer)
        return activated_layers[-1]

    def train_batch(self, batch):
        avg_error = 0.0
        dws = [np.zeros(w.shape) for w in self.ws]
        dbs = [np.zeros(b.shape) for b in self.bs]

        # Backwards propgation, adjusting dws and dbs each time
        for test_case in batch:
            raw_layers, activated_layers = self.forward_pass(test_case[0])

            avg_error += self.final_layer_handler.total_error(activated_layers[-1], test_case[1])
            
            combined_derivative = self.final_layer_handler.combined_derivative(raw_layers[-1], activated_layers[-1], test_case[1])
            for i in range(self.n_layers - 2, -1, -1):

                dws[i] -= combined_derivative @ activated_layers[i].transpose()
                dbs[i] -= combined_derivative

                if i != 0:
                    diff_prev_layer = self.ws[i].transpose() @ combined_derivative
                    combined_derivative = self.activation.derivative(raw_layers[i]) * diff_prev_layer

        for i in range(self.n_layers - 1):        
            self.ws[i] += (self.learning_rate / len(batch)) * dws[i]
            self.bs[i] += (self.learning_rate / len(batch)) * dbs[i]

        return avg_error / len(batch)

    def train(self, training_data, log=True):
        start = time()
        random.shuffle(training_data)
        training_data = util.chunk(training_data, self.batch_size)
        for epoch, batch in enumerate(training_data):
            error = self.train_batch(batch)
            if log:
                print("Epoch:", epoch, "Error:", error)

        if log:
            print('Training took {0:.2f} seconds'.format(int(time() - start)))


    def evaluate(self, test_data, log=True):
        correct = 0
        for test_case in test_data:
            if util.classify(self.run(test_case[0])) == util.classify(test_case[1]):
                correct += 1
        error = 1 - (correct / len(test_data))

        if log:
            print('Network has an error of {:.2%}'.format(error))






