import numpy as np
from layers import *

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
        #COMPUTE LOSS: Using simplification. So it only works
        # if you use one of the following pairs:
        #   Output activation   | Loss Function
        # --------------------------------------
        #       identity        | Squared Error
        #   logistic sigmoid    | Binary Cross Entropy
        #       softmax         | Multiclass Cross Entropy
        dA_prev = y_hat - y

        for idx, layer in reversed(list(enumerate(self.network))):
            dA_curr = dA_prev
            A_prev = self.memory[idx-len(self.network)]['A']
            Z_curr = self.memory[idx-len(self.network)]['Z']
            W_curr = self.params[idx]['W']
            activation = self.architecture[idx]['activation']

            dA_prev, dW_curr, db_curr = layer.backward(
                dA_curr, W_curr, Z_curr, A_prev, activation)
            
            self.gradients.append({'dW':dW_curr, 'db':db_curr})

    def _gradient_descent(self, learn_rate):
        for idx, layer in enumerate(self.network):
            self.params[idx]['W'] -= learn_rate * list(reversed(self.gradients))[idx-len(self.network)]['dW'].T
            self.params[idx]['b'] -= learn_rate * list(reversed(self.gradients))[idx-len(self.network)]['db']

    def _get_accuracy(self, y_hat, y):
        ''' y_hat and y are onehot'''
        pred = np.argmax(y_hat, axis=1)
        pred = self._one_hot_encode(pred)
        #print(pred.shape, y_hat.shape, y.shape)
        #acc = np.sum(np.where(pred == y, 1, 0), axis=0)
        #print(np.array(pred==y).shape)
        return np.mean(pred == y)

    def _calculate_loss(self, y_hat, y):
        ''' Cross Entropy Loss '''
        #eps = 1e-5
        #log_prob = -np.log(y_hat + eps)
        #return np.sum(log_prob)/len(y)
        #BINARY CROSS ENTROPY RIGHT NOW
        logits = np.multiply(np.log(y_hat), y) + np.multiply((1 - y), np.log(1 - y_hat))
        return - (np.sum(logits) / len(y))

    def _one_hot_encode(self, y):
        y = y.astype(int)
        num_classes = self.architecture[-1]['output_dim']
        return np.eye(num_classes)[y]

    def norm(self, X, x_min, x_max):
        nom = (X-X.min(axis=0))*(x_max-x_min)
        denom = X.max(axis=0) - X.min(axis=0)
        denom[denom==0] = 1
        return x_min + nom/denom 
    
    def fit(self, X, y, init_range, epochs, learn_rate):
        ''' Train the NN 
        TODO - Add in minibatching
             - Add functionality to evaluate on dev/validation set
             - Add Cross-Validation
             - Add in regression functionality
             - Be able to predict on test set
        '''
        self._init_layers(X.shape[1], init_range)
        X = self.norm(X,0,1)
        y = self._one_hot_encode(y)

        self.loss = []
        #self.accuracy = []

        while self.curr_epoch <= epochs:
            y_hat = self._forward_propagation(X)

            self.loss.append(self._calculate_loss(y_hat, y))
            #self.accuracy.append(self._get_accuracy(y, y_hat))

            self._backward_propagation(y_hat, y)
            self._gradient_descent(learn_rate)
        
            if self.curr_epoch % 500 == 0:
                s = 'EPOCH: {}, LOSS: {}'.format(self.curr_epoch, self.loss[-1])
                print(s)

            self.curr_epoch += 1
    
    def evaluate(self, X, y):
        y = self._one_hot_encode(y)
        y_hat = self._forward_propagation(X)
        #preds = y_hat > 0.5
        preds = np.rint(y_hat)
        #print(preds.shape, y.shape)
        #print(np.concatenate((preds,y),axis=1))
        return np.mean(preds == y)
