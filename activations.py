"""
Available activation functions with their respective derivatives.
The current options are:
    Sigmoid
    tanh (hyperbolic tangent)
    ReLU (Rectified Linear Unit)
"""
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    return np.exp(-x) / (1 + np.exp(-x))**2

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1 - np.tanh(x)**2

def relu(x):
    return np.maximum(x, 0)

def relu_prime(x):
    return np.array(x >= 0).astype('int')