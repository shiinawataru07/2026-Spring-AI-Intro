import mnist
from copy import deepcopy
from typing import List
from autograd.BaseGraph import Graph
from autograd.utils import buildgraph
from autograd.BaseNode import *

# 超参数
# TODO: You can change the hyperparameters here
lr = 2e-5   # 学习率
wd1 = 1e-5  # L1正则化
wd2 = 1e-5  # L2正则化
batchsize = 128

def buildGraph(Y):
    """
    建图
    @param Y: n 样本的label
    @return: Graph类的实例, 建好的图
    """
    # TODO: YOUR CODE HERE
    feats = mnist.trn_X.shape[1]
    graph = [Linear(feats, 1024), 
             relu(),
             Linear(1024, 512),
             relu(),
             Dropout(0.1),
             Linear(512, 256),
             relu(),
             Dropout(0.1),
             Linear(256, mnist.num_class),
             Softmax(),
             CrossEntropyLoss(Y)]
    return Graph(graph)
