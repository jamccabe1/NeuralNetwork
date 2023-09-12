import numpy as np


class DenseLayer:
    def __init__(self, neurons):
        self.neurons = neurons

    def _relu(self, Z):
        return np.maximum(0,Z)

    def _sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def _softmax(self, Z):
        
        exp = np.exp(Z)
        exp_sum = np.sum(exp, axis=1, keepdims=True)
        return exp/exp_sum

    def _derivative_relu(self, dA, Z):
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        return dZ

    def _derivative_sigmoid(self, dA, Z):
        sig = self._sigmoid(Z)
        return dA*sig*(1-sig)
        
    def forward(self, A_prev, W_curr, b_curr, activation):
        ''' Forward Propagation of a single layer '''
        Z_curr = np.dot(A_prev, W_curr.T) + b_curr
        if activation == "relu":
            A_curr = self._relu(Z_curr)
        elif activation == "sigmoid":
            A_curr = self._sigmoid(Z_curr)
        elif activation == "softmax":
            A_curr = self._softmax(Z_curr)
        else:
            raise Exception("Activation function "+activation+ " not supported")
        
        return A_curr, Z_curr

    def backward(self, dA_curr, W_curr, Z_curr, A_prev, activation):
        ''' Backward Propagation of a single layer '''
        m = A_prev.shape[0]

        if activation == 'softmax':
            dW_curr = (1/m) * np.dot(A_prev.T, dA_curr)
            db_curr = (1/m) * np.sum(dA_curr, axis=0, keepdims=True)
            dA_prev = np.dot(dA_curr, W_curr)
        elif activation == 'sigmoid':
            dW_curr = (1/m) * np.dot(A_prev.T, dA_curr)
            db_curr = (1/m) * np.sum(dA_curr, axis=0, keepdims=True)
            dA_prev = np.dot(dA_curr, W_curr)
        elif activation == 'relu':
            dZ_curr = self._derivative_relu(dA_curr, Z_curr)
            dW_curr = (1/m) * np.dot(A_prev.T, dZ_curr)
            db_curr = (1/m) * np.sum(dZ_curr, axis=0, keepdims=True)
            dA_prev = np.dot(dZ_curr, W_curr)
        else:
            raise Exception('Activation function '+activation+' not supported')

        return dA_prev, dW_curr, db_curr
