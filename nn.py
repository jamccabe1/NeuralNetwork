import numpy as np
import argparse as ap

class NeuralNetwork:
    #def __init__(self, args, architecture):
    def __init__(self, architecture):
        self.epochs = 30 #args.epochs
        self.learn_rate = 0.01 #args.learn_rate
        self.type = 'C' #args.type
        self.init_range = 1 #args.init_range
        #self.mb = args.mb
        self.architecture = architecture
        self.params = self.init_layers()
        self.memory = {}
        self.gradients = {}

    def init_layers(self):
        np.random.seed(99)
        params = {}

        for idx, layer in enumerate(self.architecture):
            layer_idx = idx + 1
            input_size = layer['input_dim']
            output_size = layer['output_dim']

            params['W'+str(layer_idx)] = np.random.uniform(
                -self.init_range, self.init_range, (output_size, input_size))
            params['b'+str(layer_idx)] = np.random.uniform(
                -self.init_range, self.init_range, (output_size, input_size))

        return params

    def forward_one_layer(A_prev, W_curr, b_curr, activation):
        Z_curr = np.dot(W_curr, A_prev) + b_curr

        if activation == "relu":
            hidden_activation = relu
        elif activation == "sigmoid":
            hidden_activation = sigmoid
        else:
            raise Exception("Activation function "+activation+ " not supported")
        
        return hidden_activation(Z_curr), Z_curr
    
    def forward_propagation(self, X):
        A_curr = X

        for idx, layer in enumerate(self.architecture):
            layer_idx = idx + 1
            A_prev = A_curr
            
            activation = layer['activation']
            W_curr =  self.params['W'+str(layer_idx)]
            b_curr = self.params['b'+str(layer_idx)]

            A_curr, Z_curr = self.forward_one_layer(A_prev, W_curr, b_curr, activation)

            self.memory['A'+str(idx)] = A_prev
            self.memory['Z'+str(layer_idx)] = Z_curr

        return A_curr
    
    def backward_one_layer(dA_curr, W_curr, Z_curr, A_prev, activation):
        m = A_prev.shape[1]

        if activation == 'relu':
            backward_hidden_activation = relu_backward
        elif activation == 'sigmoid':
            backward_hidden_activation = sigmoid_backward
        else:
            raise Exception('Activation function '+activation+' not supported')
        
        dZ_curr = backward_hidden_activation(dA_curr, Z_curr)
        dW_curr = np.dot(dZ_curr, A_prev.T) / m
        db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
        dA_prev = np.dot(W_curr.T, dZ_curr)

        return dA_prev, dW_curr, db_curr
    
    def backward_propagation(self, y_hat, y):
        m = y.shape[1]
        y = y.reshape(y_hat.shape)

        dA_prev = -(np.divide(y, y_hat) - np.divide(1-y, 1-y_hat))

        for prev_layer_idx, layer in reversed(list(enumerate(self.architecture))):
            curr_layer_idx = prev_layer_idx + 1
            hidden_activation = layer['activation']

            dA_curr = dA_prev
            A_prev = self.memory['A'+str(prev_layer_idx)]
            Z_curr = self.memory['Z'+str(curr_layer_idx)]
            W_curr = self.params['W'+str(curr_layer_idx)]
            #b_curr = self.params['b'+str(curr_layer_idx)]

            dA_prev, dW_curr, db_curr = self.backward_one_layer(
                dA_curr, W_curr, Z_curr, A_prev, hidden_activation)
            
            self.gradients['dW'+str(curr_layer_idx)] = dW_curr
            self.gradients['db'+str(curr_layer_idx)] = db_curr

    
    def update_weights(self):
        for idx, layer in enumerate(self.architecture):
            self.params['W'+str(idx)] -= self.learn_rate * self.gradients['dW'+str(idx)]
            self.params['b'+str(idx)] -= self.learn_rate * self.gradients['db'+str(idx)]
    

    def fit(self, X, y):
        '''
        Training Loop:

        w = init_weights()
        for each epoch e:
            for each mb_x, mb_y in e:
                predictions = h(mb_x, w) #forward prop
                loss = compute_loss(predictions, mb_y)
                gradients = back_prop(loss, w) #or could be back_prop(mb_y,preds,y)
                w = update_weights(grads, w, learn_rate)
                if verbose:
                    eval on dev
                    if best dev result yet:
                        checkpoint(w)
                    if too long since best:
                        early stopping
            eval on dev if not verbose
        '''
        history = []
        if self.type == 'C':
            loss = self.cross_entropy
        elif self.type == 'R':
            loss = self.mean_squared_error
        else:
            raise Exception('Problem type not supported')

        for i in range(self.epochs):
            y_hat = self.forward_propagation(X)
            history.append(loss(y_hat, y))
            self.backward_propagation(y_hat, y)
            self.update_weights()

        return self.params, history

    def mean_squared_error(y_hat, y):
        diff = np.abs(y_hat - y)
        return np.mean(np.square(diff))
    
    def cross_entropy(y_hat, y):
        y_softmax = softmax(y_hat)
        loss = 0

        for i, j in zip(y_softmax, y):
            loss = loss + (-1 * j * np.log(i))

        return loss





def defineFlags():
    '''
    Build the commandline argument parser
    '''
    parser = ap.ArgumentParser()
    parser.add_argument('-v',"--VERBOSE", default=False)
    parser.add_argument('train_feat',    help='the name of training set feature fileThe file should contain N lines (where N is the number of data points), and each line should contains D space-delimited floating point values (where D is the feature dimension).')
    parser.add_argument('train_target',  help='the name of the training set target (label) file. If PROBLEM_MODE (see below) is C (for classification) this should be a file with N lines, where each line contains a single integer in the set {0, 1, . . . , C −1}indicating the class label. If PROBLEM_MODE is R (for regression), this should be a file with N lines, where each line contains C space-delimited floating point values. In either case, this file contains the true outputs for all N datapoints.')
    parser.add_argument('dev_feat',      help='the name of the development set feature file, in the same format as TRAIN_FEAT_FN.')
    parser.add_argument('dev_target',    help='the name of the development set target (label) file, in the same format as TRAIN_TARGET_FN.')
    parser.add_argument('epochs',        type=int, help='the total number of epochs (i.e. passes through the data) to train for')
    parser.add_argument('learn_rate',     type=float, help='the step size to use for training (with MB-SGD)')
    #parser.add_argument('nunits',        type=int,help='the dimension of the hidden layers (aka number of hidden units per hidden layer). All hidden layers will have this same size.')
    parser.add_argument('type',          help='this should be either C (to indicate classification) or R (to indicate regresion).')
    #parser.add_argument('hidden_act',    help='this is the element-wise, non-linear function to apply at each hidden layer, and can be sig (for logistic sigmoid), tanh (for hyperbolic tangent) or relu (for rectified linear unit).')
    parser.add_argument('init_range',    type=float,help='all of your weights (including bias vectors) should be initialized uniformly random in the range [−INIT_RANGE, INIT_RANGE]).')
    #parser.add_argument('num_classes',   type=int,help='The number of classes (for classification) or the dimension of the output vector (if regression).')
    #parser.add_argument('mb',type=int,help='this specifies the number of data points to be included in each mini-batch.')
    #parser.add_argument('-nlayers','--NUM_HIDDEN_LAYERS', type=int,default=1,help='this is the number of hidden layers in your neural network.')
    return parser

def loadData(args):
    train_X = np.loadtxt(args.train_feat, dtype=float).astype(np.float32)
    train_y = np.loadtxt(args.train_target, dtype=float).astype(np.float32)
    dev_X = np.loadtxt(args.dev_feat, dtype=float).astype(np.float32)
    dev_y = np.loadtxt(args.dev_target, dtype=float).astype(np.float32)
    return train_X, train_y, dev_X, dev_y

def relu(Z):
    return np.maximum(0,Z)

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def softmax(Z):
    exp_Z = np.exp(Z)
    exp_Z_sum = np.sum(exp_Z)
    return exp_Z/exp_Z_sum

def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def sigmoid_backward(dA, Z):
    sig = sigmoid(Z)
    return dA*sig*(1-sig)


#args = defineFlags().parse_args()
#train_X, train_y, dev_X, dev_y = loadData(args)

train_X = np.loadtxt("./prog1_data/dataset11.train_features.txt", dtype=float).astype(np.float32)
train_y = np.loadtxt("./prog1_data/dataset11.train_targets.txt", dtype=float).astype(np.float32)
dev_X = np.loadtxt("./prog1_data/dataset11.dev_features.txt", dtype=float).astype(np.float32)
dev_y = np.loadtxt("./prog1_data/dataset11.dev_targets.txt", dtype=float).astype(np.float32)

architecture = [
    {'input_dim':8,  'output_dim':10, 'activation':'relu'},
    {'input_dim':10, 'output_dim':2,  'activation':'relu'},
    {'input_dim':2,  'output_dim':1,  'activation':'sigmoid'} 
]

#neuralnet = NeuralNetwork(args, architecture)
neuralnet = NeuralNetwork(architecture)
params, history = neuralnet.fit(train_X, train_y)