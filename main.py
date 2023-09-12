import numpy as np
import argparse as ap

from network import *
from layers import *



            

train_X = np.loadtxt("./data/dataset9.train_features.txt", dtype=float).astype(np.float32)
train_y = np.loadtxt("./data/dataset9.train_targets.txt", dtype=float).astype(np.float32)
dev_X   = np.loadtxt("./data/dataset9.dev_features.txt", dtype=float).astype(np.float32)
dev_y   = np.loadtxt("./data/dataset9.dev_targets.txt", dtype=float).astype(np.float32)

model = Network()
model.add(DenseLayer(50))
model.add(DenseLayer(50))
model.add(DenseLayer(50))
model.add(DenseLayer(25))
model.add(DenseLayer(2))
model.compile(train_X)
model.fit(train_X, train_y, 0.01, 1, 0.001)

accuracy = model.evaluate(dev_X, dev_y)
print(accuracy)
#print(dev_y)