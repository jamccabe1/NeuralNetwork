import numpy as np
import argparse as ap

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
        if activation == 'softmax':
            dW_curr = np.dot(A_prev.T, dA_curr)
            db_curr = np.sum(dA_curr, axis=0, keepdims=True)
            dA_prev = np.dot(dA_curr, W_curr)
        elif activation == 'relu':
            dZ_curr = self._derivative_relu(dA_curr, Z_curr)
            dW_curr = np.dot(A_prev.T, dZ_curr)
            db_curr = np.sum(dZ_curr, axis=0, keepdims=True)
            dA_prev = np.dot(dZ_curr, W_curr)
        else:
            raise Exception('Activation function '+activation+' not supported')

        return dA_prev, dW_curr, db_curr


class Network:
    def __init__(self):
        self.curr_epoch = 0
        self.network = [] # layers
        self.architecture = [] # mapping input neurons --> output neurons
        self.params = [] # W, b
        self.memory = [] # Z, A
        self.gradients = [] # dW, db
        self.loss = [] # training set loss
        self.dev_loss = [] # dev set loss

    def add(self, layer, activation='relu'):
        ''' Add a layer to the NN '''
        self.network.append(layer)

    def compile(self, X):
        ''' Initialize the model's architecture '''
        for idx, layer in enumerate(self.network):
            if idx == 0:
                self.architecture.append({'input_dim':X.shape[1],
                                  'output_dim':self.network[0].neurons,
                                  'activation':'relu'})
            elif 0 < idx and idx < len(self.network) - 1:
                self.architecture.append({'input_dim':self.network[idx-1].neurons,
                                  'output_dim':self.network[idx].neurons,
                                  'activation':'relu'})
            else:
                self.architecture.append({'input_dim':self.network[idx-1].neurons,
                                  'output_dim':self.network[idx].neurons,
                                  'activation':'softmax'})
                   

    def _init_layers(self, num_data, init_range):
        ''' Initialize the weights and biases of each layer '''
        np.random.seed(99)
        
        for idx, layer in enumerate(self.architecture):
            input_size = layer['input_dim']
            output_size = layer['output_dim']

            self.params.append({
                'W':np.random.uniform(-init_range, init_range, (output_size, input_size)),
                'b':np.random.uniform(-init_range, init_range, (1, output_size))
            })
            #self.memory.append({
            #    'Z':np.ones((num_data, output_size)),
            #    'A':np.ones((num_data, input_size))
            #})
            #self.gradients.append({
            #    'dW':np.ones((output_size, input_size)),
            #    'db':np.ones((1, output_size))
            #})
            

    def _forward_propagation(self, X):
        ''' Does a forward pass through full NN '''
        A_curr = X

        for idx in range(len(self.params)):
            A_prev = A_curr
            W_curr = self.params[idx]['W']
            b_curr = self.params[idx]['b']
            activation = self.architecture[idx]['activation']
            A_curr, Z_curr = self.network[idx].forward(A_prev,
                                                     W_curr,
                                                     b_curr,
                                                     activation)
            #self.memory[idx] = {'A':A_prev , 'Z':Z_curr}
            self.memory.append({'A':A_prev, 'Z':Z_curr})
    
        return A_curr

    def _backward_propagation(self, y_hat, y):
        ''' Does a backward pass through full NN '''
        #y = y.reshape(y_hat.shape)
        #COMPUTE LOSS: Only giong to do categorical cross-entropy
        #              for multi-classification first
        #dA_prev = -(np.divide(y, y_hat) - np.divide(1-y, 1-y_hat))
        dA_prev = y_hat - 1
        dA_prev /= len(y)

        for idx, layer in reversed(list(enumerate(self.network))):
            #mem_idx = -(idx +1)
            #print('idx:',idx,'len(memory)',len(self.memory))
            dA_curr = dA_prev
            A_prev = self.memory[idx]['A']
            Z_curr = self.memory[idx]['Z']
            W_curr = self.params[idx]['W']
            activation = self.architecture[idx]['activation']

            dA_prev, dW_curr, db_curr = layer.backward(
                dA_curr, W_curr, Z_curr, A_prev, activation)
            
            #self.gradients[idx] = {'dW':dW_curr , 'db':db_curr}
            self.gradients.append({'dW':dW_curr, 'db':db_curr})

    def _gradient_descent(self, learn_rate):
        for idx, layer in enumerate(self.network):
            self.params[idx]['W'] -= learn_rate * list(reversed(self.gradients))[idx]['dW'].T
            self.params[idx]['b'] -= learn_rate * list(reversed(self.gradients))[idx]['db']

    def _get_accuracy(self, y_hat, y):
        return np.mean(np.argmax(y_hat, axis=1) == y)
        
    def _calculate_loss(self, y_hat, y):
        ''' Cross Entropy Loss '''
        eps = 1e-5
        log_prob = -np.log(y_hat + eps)
        return np.sum(log_prob)/len(y)

    def fit(self, X, y, init_range, epochs, learn_rate):
        ''' Train the NN 
        TODO - Add in minibatching
             - Add functionality to evaluate on dev/validation set
             - Add Cross-Validation
             - Add in regression functionality
             - Be able to predict on test set
        '''
        self._init_layers(X.shape[1], init_range)
        
        self.loss = []
        self.dev_loss = []
        self.accuracy = []
        while self.curr_epoch <= epochs:
            y_hat = self._forward_propagation(X)

            self.loss.append(self._calculate_loss(y_hat, y))
            self.accuracy.append(self._get_accuracy(y_hat, y))

            self._backward_propagation(y_hat, y)
            self._gradient_descent(learn_rate)

        
            if self.curr_epoch % 100 == 0:
                s = 'EPOCH: {}, ACCURACY: {}, LOSS: {}'.format(self.curr_epoch, self.accuracy[-1], self.loss[-1])
                print(s)

            self.curr_epoch += 1
            


train_X = np.loadtxt("./data/dataset2.train_features.txt", dtype=float).astype(np.float32)
train_y = np.loadtxt("./data/dataset2.train_targets.txt", dtype=float).astype(np.float32)
dev_X   = np.loadtxt("./data/dataset2.dev_features.txt", dtype=float).astype(np.float32)
dev_y   = np.loadtxt("./data/dataset2.dev_targets.txt", dtype=float).astype(np.float32)

model = Network()
model.add(DenseLayer(4))
#model.add(DenseLayer(6))
#model.add(DenseLayer(8))
#model.add(DenseLayer(4))
model.add(DenseLayer(3))
model.compile(train_X)
#model._init_layers(1)
model.fit(train_X, train_y, 0.4, 50, 0.1)
print('gradients length: ', len(model.gradients))
print('params length: ', len(model.params))
print('memory length: ', len(model.memory))
print(train_X.shape)
print('W1',model.params[0]['W'].shape, ' b1', model.params[0]['b'].shape)
print('Z1', model.memory[0]['Z'].shape,' A1', model.memory[0]['A'].shape)
print('dW1',model.gradients[1]['dW'].shape, ' db1', model.gradients[1]['db'].shape)
print()
print('W2',model.params[1]['W'].shape, ' b2', model.params[1]['b'].shape)
print('Z2',model.memory[1]['Z'].shape, ' A2',model.memory[1]['A'].shape)
print('dW2',model.gradients[0]['dW'].shape, ' db2', model.gradients[0]['db'].shape)