from typing import List
import math
import numpy as np
from .Init import * 

def shape(X):
    if isinstance(X, np.ndarray):
        ret = "ndarray"
        if np.any(np.isposinf(X)):
            ret += "_posinf"
        if np.any(np.isneginf(X)):
            ret += "_neginf"
        if np.any(np.isnan(X)):
            ret += "_nan"
        return f" {X.shape} "
    if isinstance(X, int):
        return "int"
    if isinstance(X, float):
        ret = "float"
        if np.any(np.isposinf(X)):
            ret += "_posinf"
        if np.any(np.isneginf(X)):
            ret += "_neginf"
        if np.any(np.isnan(X)):
            ret += "_nan"
        return ret
    else:
        raise NotImplementedError(f"unsupported type {type(X)}")

class Node(object):
    def __init__(self, name, *params):
        self.grad = [] # 节点的梯度，self.grad[i]对应self.params[i]在反向传播时的梯度
        self.cache = [] # 节点保存的临时数据
        self.name = name # 节点的名字
        self.params = list(params) # 用于Linear节点中存储weight和bias参数使用

    def num_params(self):
        return len(self.params)

    def cal(self, X):
        '''
        计算函数值。请在其子类中完成具体实现。
        '''
        pass

    def backcal(self, grad):
        '''
        计算梯度。请在其子类中完成具体实现。
        '''
        pass

    def flush(self):
        '''
        初始化或刷新节点内部数据，包括梯度和缓存
        '''
        self.grad = []
        self.cache = []

    def forward(self, X, debug=False):
        '''
        正向传播。输入X，输出正向传播的计算结果。
        '''
        if debug:
            print(self.name, shape(X))
        ret = self.cal(X)
        if debug:
            print(shape(ret))
        return ret

    def backward(self, grad, debug=False):
        '''
        反向传播。输入grad（该grad为反向传播到该节点的梯度），输出反向传播到下一层的梯度。
        '''
        if debug:
            print(self.name, shape(grad))
        ret = self.backcal(grad)
        if debug:
            print(shape(ret))
        return ret
    
    def eval(self):
        pass

    def train(self):
        pass


class relu(Node):
    # input X: (*)，即可能是任意维度
    # output relu(X): (*)
    def __init__(self):
        super().__init__("relu")

    def cal(self, X):
        self.cache.append(X)
        return np.clip(X, 0, None)

    def backcal(self, grad):
        return np.multiply(grad, self.cache[-1] > 0) 

class sigmoid(Node):
    # input X: (*)，即可能是任意维度
    # output sigmoid(X): (*)
    def __init__(self):
        super().__init__("sigmoid")

    def cal(self, X):
        # TODO: YOUR CODE HERE
        ret = np.where(X>=0, 1/(1+np.exp(-X)), np.exp(X)/(1+np.exp(X)))
        self.cache.append(ret)
        return ret

    def backcal(self, grad):
        # TODO: YOUR CODE HERE
        sigmoidX = self.cache[-1]
        return np.multiply(grad, np.multiply(sigmoidX, 1-sigmoidX))
    
class tanh(Node):
    # input X: (*)，即可能是任意维度
    # output tanh(X): (*)
    def __init__(self):
        super().__init__("tanh")

    def cal(self, X):
        ret = np.tanh(X)
        self.cache.append(ret)
        return ret


    def backcal(self, grad):
        return np.multiply(grad, np.multiply(1+self.cache[-1], 1-self.cache[-1]))
    

class Linear(Node):
    # input X: (*,d1)
    # param weight: (d1, d2)
    # param bias: (d2)
    # output Linear(X): (*, d2)
    def __init__(self, indim, outdim):
        """
        初始化
        @param indim: 输入维度
        @param outdim: 输出维度
        """
        weight = kaiming_uniform(indim, outdim)
        bias = zeros(outdim)
        super().__init__("linear", weight, bias)

    def cal(self, X):
        # TODO: YOUR CODE HERE
        self.cache.append(X)
        return np.dot(X, self.params[0]) + self.params[1]

    def backcal(self, grad):
        '''
        需要保存weight和bias的梯度，可以参考Node类和BatchNorm类
        '''
        # TODO: YOUR CODE HERE
        X = self.cache[-1]
        self.grad.append(np.dot(X.T, grad))
        self.grad.append(grad.sum(axis=0))
        return np.dot(grad, self.params[0].T)


class StdScaler(Node):
    '''
    input shape (*)
    output (*)
    '''
    EPS = 1e-3
    def __init__(self, mean, std):
        super().__init__("StdScaler")
        self.mean = mean
        self.std = std

    def cal(self, X):
        X = X.copy()
        X -= self.mean
        X /= (self.std + self.EPS)
        return X

    def backcal(self, grad):
        return grad/ (self.std + self.EPS)
    


class BatchNorm(Node):
    '''
    input shape (*)
    output (*)
    '''
    EPS = 1e-8
    def __init__(self, indim, momentum: float = 0.9):
        super().__init__("batchnorm", ones((indim)), zeros(indim))
        self.momentum = momentum
        self.mean = None
        self.std = None
        self.updatemean = True
        self.indim = indim

    def cal(self, X):
        if self.updatemean:
            tmean, tstd = np.mean(X, axis=0, keepdims=True), np.std(X, axis=0, keepdims=True)
            if self.mean is None or self.std is None:
                self.mean = tmean
                self.std = tstd
            else:
                self.mean *= self.momentum
                self.mean += (1-self.momentum) * tmean
                self.std *= self.momentum
                self.std += (1-self.momentum) * tstd
        X = X.copy()
        X -= self.mean
        X /= (self.std + self.EPS)
        self.cache.append(X.copy())
        X *= self.params[0]
        X += self.params[1]
        return X

    def backcal(self, grad):
        X = self.cache[-1]
        self.grad.append(np.multiply(X, grad).reshape(-1, self.indim).sum(axis=0))
        self.grad.append(grad.reshape(-1, self.indim).sum(axis=0))
        return (grad*self.params[0])/ (self.std + self.EPS)
    
    def eval(self):
        self.updatemean = False

    def train(self):
        self.updatemean = True


class Dropout(Node):
    '''
    input shape (*)
    output (*)
    '''
    def __init__(self, p: float = 0.1):
        super().__init__("dropout")
        assert 0<=p<=1, "p 是dropout 概率，必须在[0, 1]中"
        self.p = p
        self.dropout = True

    def cal(self, X):
        if self.dropout:
            X = X.copy()
            mask = np.random.rand(*X.shape) < self.p
            np.putmask(X, mask, 0)
            X = X * (1/(1-self.p))
            self.cache.append(mask)
        return X
    
    def backcal(self, grad):
        if self.dropout:
            grad = grad.copy()
            np.putmask(grad, self.cache[-1], 0)
            grad = grad * (1/(1-self.p))
        return grad
    
    def eval(self):
        self.dropout=False

    def train(self):
        self.dropout=True


class Softmax(Node):
    # input X: (*)
    # output softmax(X): (*), softmax at 'dim'
    def __init__(self, dim=-1):
        super().__init__("softmax")
        self.dim = dim

    def cal(self, X):
        X = X - np.max(X, axis=self.dim, keepdims=True)
        expX = np.exp(X)
        ret = expX / expX.sum(axis=self.dim, keepdims=True)
        self.cache.append(ret)
        return ret

    def backcal(self, grad):
        softmaxX = self.cache[-1]
        grad_p = np.multiply(grad, softmaxX)
        return grad_p - np.multiply(grad_p.sum(axis=self.dim, keepdims=True), softmaxX)


class LogSoftmax(Node):
    # input X: (*)
    # output logsoftmax(X): (*), logsoftmax at 'dim'
    def __init__(self, dim=-1):
        super().__init__("logsoftmax")
        self.dim = dim

    def cal(self, X):
        # TODO: YOUR CODE HERE
        EPS = 1e-6
        X = X - np.max(X, axis=self.dim, keepdims=True)
        expX = np.exp(X)
        ret = np.log(expX / expX.sum(axis=self.dim, keepdims=True) + EPS)
        self.cache.append(ret)
        return ret

    def backcal(self, grad):
        # TODO: YOUR CODE HERE
        return grad - np.multiply(grad.sum(axis=self.dim, keepdims=True), np.exp(self.cache[-1]))




class NLLLoss(Node):
    '''
    negative log-likelihood 损失函数
    '''
    # shape X: (*, d), y: (*)
    # shape value: number 
    # 输入：X: (*) 个预测，每个预测是个d维向量，代表d个类别上分别的log概率。  y：(*) 个整数类别标签
    # 输出：NLL损失
    def __init__(self, y):
        """
        初始化
        @param y: n 样本的label
        """
        super().__init__("NLLLoss")
        self.y = y

    def cal(self, X):
        y = self.y
        self.cache.append(X)
        return - np.sum(
            np.take_along_axis(X, np.expand_dims(y, axis=-1), axis=-1))

    def backcal(self, grad):
        X, y = self.cache[-1], self.y
        ret = np.zeros_like(X)
        np.put_along_axis(ret, np.expand_dims(y, axis=-1), -1, axis=-1)
        return grad * ret



class CrossEntropyLoss(Node):
    '''
    多分类交叉熵损失函数，不同于课上讲的二分类。它与NLLLoss的区别仅在于后者输入log概率，前者输入概率。
    '''
    # shape X: (*, d), y: (*)
    # shape value: number 
    # 输入：X: (*) 个预测，每个预测是个d维向量，代表d个类别上分别的概率。  y：(*) 个整数类别标签
    # 输出：交叉熵损失
    def __init__(self, y):
        """
        初始化
        @param y: n 样本的label
        """
        super().__init__("CELoss")
        self.y = y

    def cal(self, X):
        # TODO: YOUR CODE HERE
        # 提示，可以对照NLLLoss的cal
        EPS = 1e-6
        y = self.y
        logX = np.log(X + EPS)
        self.cache.append(X)
        return - np.sum(
            np.take_along_axis(logX, np.expand_dims(y, axis=-1), axis=-1))

    def backcal(self, grad):
        # TODO: YOUR CODE HERE
        # 提示，可以对照NLLLoss的backcal
        EPS = 1e-6
        X, y = self.cache[-1], self.y
        ret = np.zeros_like(X)
        np.put_along_axis(ret, np.expand_dims(y, axis=-1), -1, axis=-1)
        return grad * ret / (X + EPS)
    
class Conv2D(Node):
    # input X: (n, inchannels, h, w)
    # param weight: (outchannels, inchannels, kernel_size, kernel_size)
    # param bias: (outchannels)
    # output Conv2D(X): (n, outchannels, h_out, w_out)
    def __init__(self, inchannels, outchannels, kernel_size, stride=1, padding=0):
        fan_in = inchannels * kernel_size * kernel_size
        std = math.sqrt(2.0 / fan_in)
        weight = np.random.randn(outchannels, inchannels, kernel_size, kernel_size) * std
        super().__init__("Conv2D", weight, np.zeros(outchannels))
        self.inchannels = inchannels
        self.outchannels = outchannels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
    @staticmethod
    def im2col(X, K, stride=1, padding=0):
        # X: (B, C, H, W)
        # output im2col(X): (B, OH*OW, C*K*K)
        B, C, H, W = X.shape
        if padding > 0:
            X_pad = np.pad(X,((0, 0),(0, 0),(padding, padding),(padding, padding)))
        else:
            X_pad = X
        OH = (X_pad.shape[2] - K) // stride + 1
        OW = (X_pad.shape[3] - K) // stride + 1
        Bs, Cs, Hs, Ws = X_pad.strides
        shape = (B, OH, OW, C, K, K)
        strides = (Bs, Hs*stride, Ws*stride, Cs, Hs, Ws)
        windows = np.lib.stride_tricks.as_strided(X_pad, shape=shape, strides=strides, writeable=False)
        return windows.reshape(B, OH*OW, C*K*K)

    def cal(self, X):
        # X: (B, C, H, W)
        # cal(X): (B, OC, OH, OW)
        B, C, H, W_in = X.shape
        OC, C, K, K = self.params[0].shape
        OH = (H + 2*self.padding - K) // self.stride + 1
        OW = (W_in + 2*self.padding - K) // self.stride + 1
        W = self.params[0]
        b = self.params[1]
        W_col = W.reshape(OC, -1)
        X_col = Conv2D.im2col(X, K, self.stride, self.padding)
        self.cache.append((X, X_col))
        out = X_col @ W_col.T + b.reshape(1, 1, OC) # out: (B, OH*OW, OC)
        out = out.reshape(B, OH, OW, OC)
        return out.transpose(0, 3, 1, 2)
    
    @staticmethod
    def col2im(X_col, X_shape, K, stride=1, padding=0):
        # X_col: (B, OH*OW, C*K*K)
        # col2im(X_col): (B, C, H, W)
        B, C, H, W = X_shape
        Hp = H + 2*padding
        Wp = W + 2*padding
        X_pad = np.zeros((B, C, Hp, Wp))
        OH = (Hp - K) // stride + 1
        OW = (Wp - K) // stride + 1
        X_col = X_col.reshape(B, OH, OW, C, K, K)
        for y in range(K):
            for x in range(K):
                X_pad[:, :, y:y+OH*stride:stride, x:x+OW*stride:stride] += X_col[:, :, :, :, y, x].transpose(0, 3, 1, 2)
        if padding > 0:
            return X_pad[:, :, padding:-padding, padding:-padding]
        else:
            return X_pad

    def backcal(self, grad):
        # grad: (B, OC, OH, OW)
        # dW: (OC, C, K, K)
        # db: (OC)
        # dX: (B, C, H, W)
        X, X_col = self.cache[-1]
        B, C, H, W = X.shape
        OC, C, K, K = self.params[0].shape
        OH, OW = grad.shape[2], grad.shape[3]
        W_col = self.params[0].reshape(OC, -1) # W_col: (OC, C*K*K)
        grad_col = grad.transpose(0, 2, 3, 1).reshape(B, OH*OW, OC) # grad_col: (B, OH*OW, OC)
        dW_col = np.einsum("bno,bnd->od", grad_col, X_col) # dW_col: (OC, C*K*K)
        dW = dW_col.reshape(OC, C, K, K)
        db = grad_col.sum(axis=(0, 1)) # db: (OC)
        dX_col = np.einsum("bno,od->bnd", grad_col, W_col) # dX_col: (B, OH*OW, C*K*K)
        dX = Conv2D.col2im(dX_col, X.shape, K, self.stride, self.padding) # dX: (B, C, H, W)
        self.grad.append(dW)
        self.grad.append(db)
        return dX
    
class Flatten(Node):
    # input X: (n, d1, d2, ...)
    # output Flatten(X): (n, d1*d2*...)
    def __init__(self):
        super().__init__("flatten")

    def cal(self, X):
        self.cache.append(X.shape)
        return X.reshape(X.shape[0], -1)

    def backcal(self, grad):
        return grad.reshape(self.cache[-1])
    
class MaxPool2D(Node):
    # input X: (B, C, H, W)
    # output: (B, C, H_out, W_out)

    def __init__(self, kernel_size=2, stride=2):
        super().__init__("maxpool2d")
        self.kernel_size = kernel_size
        self.stride = stride

    def cal(self, X):
        self.cache.append(X)
        B, C, H, W = X.shape
        K = self.kernel_size
        S = self.stride
        OH = (H - K) // S + 1
        OW = (W - K) // S + 1
        windows = np.lib.stride_tricks.as_strided(
            X,
            shape=(B, C, OH, OW, K, K),
            strides=(
                X.strides[0],
                X.strides[1],
                X.strides[2] * S,
                X.strides[3] * S,
                X.strides[2],
                X.strides[3],
            ),
            writeable=False
        )
        self.windows_shape = windows.shape
        out = np.max(windows, axis=(4, 5))
        self.max_mask = (windows == out[..., None, None])
        return out

    def backcal(self, grad):
        X = self.cache[-1]
        B, C, H, W = X.shape
        K = self.kernel_size
        S = self.stride
        OH, OW = grad.shape[2], grad.shape[3]
        grad_input = np.zeros_like(X)
        grad_expand = grad[..., None, None]
        grad_windows = self.max_mask * grad_expand
        for i in range(K):
            for j in range(K):
                grad_input[:, :, i:i + OH * S:S, j:j + OW * S:S] += grad_windows[:, :, :, :, i, j]
        return grad_input
    
class BatchNorm2D(Node):
    # input X: (B, C, H, W)
    # gamma: (1, C, 1, 1)
    # beta: (1, C, 1, 1)
    # output: (B, C, H, W)
    EPS = 1e-5
    def __init__(self, channels, momentum: float = 0.9):
        super().__init__("batchnorm2d", ones((1, channels, 1, 1)), zeros((1, channels, 1, 1)))
        self.momentum = momentum
        self.mean = None
        self.std = None
        self.updatemean = True
        
    def cal(self, X):
        if self.updatemean:
            tmean, tstd = np.mean(X, axis=(0, 2, 3), keepdims=True), np.std(X, axis=(0, 2, 3), keepdims=True)
            if self.mean is None or self.std is None:
                self.mean = tmean
                self.std = tstd
            else:
                self.mean *= self.momentum
                self.mean += (1-self.momentum) * tmean
                self.std *= self.momentum
                self.std += (1-self.momentum) * tstd
        self.cache.append(X)
        X_norm = (X - self.mean) / (self.std + self.EPS)
        return X_norm * self.params[0] + self.params[1]
    
    def backcal(self, grad):
        X = self.cache[-1]
        gamma = self.params[0]
        beta = self.params[1]
        mean = np.mean(X, axis=(0, 2, 3), keepdims=True)
        std = np.std(X, axis=(0, 2, 3), keepdims=True)
        X_norm = (X - mean) / (std + self.EPS)
        N = X.shape[0] * X.shape[2] * X.shape[3]
        dgamma = np.sum(grad * X_norm, axis=(0, 2, 3), keepdims=True)
        dbeta = np.sum(grad, axis=(0, 2, 3), keepdims=True)
        self.grad.append(dgamma)
        self.grad.append(dbeta)
        dX_norm = grad * gamma
        dX = (1.0/N) * (1.0/(std + self.EPS)) * (N * dX_norm - np.sum(dX_norm, axis=(0, 2, 3), keepdims=True) 
                                                 - X_norm * np.sum(dX_norm * X_norm, axis=(0, 2, 3), keepdims=True))
        return dX