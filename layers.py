"""
Available layers to use for Neural Network. They all inherit from the 
base "Layer". The current options are:
    Dense - a fully connected layer
    Activation - does not update parameters, only applies a specified 
                activation function.
    Flatten - flattens the input
    Conv - A convolutional layer
"""

import numpy as np
from scipy import signal

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    # Compute the output Y of a layer for given input X
    def forward_propagation(self, input):
        raise NotImplementedError
    
    # Compute dE/dX for a given dE/dY and update params if needed
    def backward_propagation(self, output_err, learn_rate):
        raise NotImplementedError


class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward_propagation(self, output_err, learning_rate):
        input_err = np.dot(output_err, self.weights.T)
        weights_err = np.dot(self.input.T, output_err)

        self.weights -= learning_rate * weights_err
        self.bias -= learning_rate * output_err
        return input_err    


class Activation(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward_propagation(self, input):
        self.input = input
        self.output = self.activation(self.input)
        return self.output
    
    def backward_propagation(self, output_err, learn_rate):
        return self.activation_prime(self.input) * output_err


class Flatten(Layer):
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = input_data.flatten().reshape((1,-1))
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        return output_error.reshape(self.input.shape)


class Conv(Layer):
    def __init__(self, input_shape, kernel_shape, layer_depth):
        self.input_shape = input_shape
        self.input_depth = input_shape[2]
        self.kernel_shape = kernel_shape
        self.layer_depth = layer_depth
        self.output_shape = (input_shape[0]-kernel_shape[0]+1, input_shape[1]-kernel_shape[1]+1, layer_depth)
        
        self.weights = np.random.rand(kernel_shape[0], kernel_shape[1], self.input_depth, layer_depth) - 0.5
        self.bias = np.random.rand(layer_depth) - 0.5

    def forward_propagation(self, input):
        self.input = input
        self.output = np.zeros(self.output_shape)

        for k in range(self.layer_depth):
            for d in range(self.input_depth):
                self.output[:,:,k] += signal.correlate2d(self.input[:,:,d], self.weights[:,:,d,k], 'valid') + self.bias[k]
        
        return self.output

    def backward_propagation(self, output_err, learn_rate):
        in_error = np.zeros(self.input_shape)
        dWeights = np.zeros((self.kernel_shape[0], self.kernel_shape[1], self.input_depth, self.layer_depth))
        dBias = np.zeros(self.layer_depth)

        for k in range(self.layer_depth):
            for d in range(self.input_depth):
                in_error[:,:,d] += signal.convolve2d(output_err[:,:,k], self.weights[:,:,d,k], 'full')
                dWeights[:,:,d,k] = signal.correlate2d(self.input[:,:,d], output_err[:,:,k], 'valid')
            dBias[k] = self.layer_depth * np.sum(output_err[:,:,k])

        self.weights -= learn_rate * dWeights
        self.bias -= learn_rate * dBias
        return in_error