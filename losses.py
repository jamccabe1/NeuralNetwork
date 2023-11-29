"""
Available loss functions with their respective derivatives.
The current options are:
    Mean Squared Error (MSE)
    Sum of Squared Error (SSE)
"""

import numpy as np

# Mean Squared Error
def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size

# Sum of Squared Error
def sse(y_true, y_pred):
    return 0.5 * np.sum(np.power(y_true - y_pred, 2))

def sse_prime(y_true, y_pred):
    return y_pred - y_true