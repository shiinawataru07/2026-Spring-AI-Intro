import numpy as np
import mnist
import pickle
from autograd.utils import PermIterator
from util import setseed
from scipy.ndimage import rotate, shift
from copy import deepcopy
from typing import List
from autograd.BaseGraph import Graph
from autograd.BaseNode import *
from autograd.utils import buildgraph

import time

setseed(0) # 固定随机数种子以提高可复现性

save_path = "model/your.npy"

def buildGraph(Y):
    """
    建图
    @param Y: n 样本的label
    @return: Graph类的实例, 建好的图
    """
    graph = [Conv2D(1, 8, 3),
             relu(),
             MaxPool2D(2, 2),
             Conv2D(8, 16, 3),
             relu(),
             MaxPool2D(2, 2),
             Flatten(),
             Linear(16*5*5, 64),
             relu(),
             Dropout(0.2),
             Linear(64, 10),
             LogSoftmax(),
             NLLLoss(Y)]
    return Graph(graph)

lr = 5e-4
wd1 = 5e-4
wd2 = 5e-4
batchsize = 64

X = mnist.val_X.astype(np.float32) / 255.0
Y = mnist.val_Y

def augment(figure):
    # 784, -> 28, 28,
    figure = figure.reshape(28, 28)
    if np.random.rand() < 0.5:
        figure = rotate(figure, np.random.uniform(-20, 20), reshape=False, mode='nearest')
    if np.random.rand() < 0.5:
        figure = shift(figure, np.random.uniform(-3, 3, size=2), mode='nearest')
    figure = np.clip(figure, 0, 1)
    return figure

def batch_augment(X):
    return np.array([augment(x) for x in X])

if __name__ == "__main__":
    # print(augment(X).shape, Y.shape)
    start = time.perf_counter()
    
    graph = buildGraph(Y)
    best_train_acc = 0
    epochs = 7
    for epoch in range(1, epochs+1):
        hatys = []
        ys = []
        losss = []
        graph.train()
        dataloader = PermIterator(X.shape[0], batchsize)
        for perm in dataloader:
            tX = batch_augment(X[perm])
            tX = tX.reshape(-1, 1, 28, 28)
            tY = Y[perm]
            graph[-1].y = tY
            graph.flush()
            pred, loss = graph.forward(tX)[-2:]
            hatys.append(np.argmax(pred, axis=1))
            ys.append(tY)
            graph.backward()
            graph.optimstep(lr, wd1, wd2)
            losss.append(loss)
        loss = np.average(losss)
        acc = np.average(np.concatenate(hatys)==np.concatenate(ys))
        print(f"epoch {epoch} loss {loss:.3e} acc {acc:.4f}")
        if acc > best_train_acc:
            best_train_acc = acc
            with open(save_path, "wb") as f:
                pickle.dump(graph, f)
        lr *= 0.95
                
    end = time.perf_counter()
    print(f"Training time: {end - start:.2f} seconds")