import numpy as np
import argparse as ap
import sys



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
    parser.add_argument('learnrate',     type=float, help='the step size to use for training (with MB-SGD)')
    parser.add_argument('nunits',        type=int,help='the dimension of the hidden layers (aka number of hidden units per hidden layer). All hidden layers will have this same size.')
    parser.add_argument('type',          help='this should be either C (to indicate classification) or R (to indicate regresion).')
    parser.add_argument('hidden_act',    help='this is the element-wise, non-linear function to apply at each hidden layer, and can be sig (for logistic sigmoid), tanh (for hyperbolic tangent) or relu (for rectified linear unit).')
    parser.add_argument('init_range',    type=float,help='all of your weights (including bias vectors) should be initialized uniformly random in the range [−INIT_RANGE, INIT_RANGE]).')
    parser.add_argument('num_classes',   type=int,help='The number of classes (for classification) or the dimension of the output vector (if regression).')
    parser.add_argument('-mb','--MINIBATCH_SIZE',type=int, default=0,help='this specifies the number of data points to be included in each mini-batch.')
    parser.add_argument('-nlayers','--NUM_HIDDEN_LAYERS', type=int,default=0,help='this is the number of hidden layers in your neural network.')
    return parser

def loadData(args):
    train_X = np.loadtxt(args.train_feat, dtype=float).astype(np.float32)
    train_y = np.loadtxt(args.train_target, dtype=float).astype(np.float32)
    dev_X = np.loadtxt(args.dev_feat, dtype=float).astype(np.float32)
    dev_y = np.loadtxt(args.dev_target, dtype=float).astype(np.float32)
    return train_X, train_y, dev_X, dev_y

args = defineFlags().parse_args()
train_X, train_y, dev_X, dev_y = loadData(args)
D = train_X.shape[1]
L = args.nunits
C = args.num_classes

print(train_X.shape[0])
print(train_X.shape[1])
print(train_y.shape[0])
#print(train_y.shape[1])

