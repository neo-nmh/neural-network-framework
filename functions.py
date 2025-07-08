import numpy as np
import math
from neuralnetwork import CLASSCOUNT

class ReLu:
    def forward(activations):
        return np.maximum(0, activations)

    def backward(weightedSum):
        if weightedSum > 0:
            return 1
        else:
            return 0


class Sigmoid:
    def forward(activations):
        return np.float32(1 / (1 + math.exp(-1 * activations)))

    def backward(weightedSum):
        sigmoid = 1 / (1 + math.exp(-1 * weightedSum))
        return np.float32(sigmoid * (1 - sigmoid))


class Softmax:
    def forward(activations):
        return

    def backward(weightedSum):
        return


class Loss:
    # 1/2 MSE loss function
    def forward(activations, label):
        loss = 0
        label = np.array([label])
        for i in range(len(activations)):
            loss += (np.square(label[i] - activations[i]))
        loss /= (2 * len(activations))
        return loss

    # dL/da = -1/n(y-a)
    def backward(activations, label):
        label = np.array([label])
        grad = np.empty(CLASSCOUNT, dtype=np.float32)
        for i in range(CLASSCOUNT):
            grad[i] = (-1 / CLASSCOUNT) * (label[i] - activations[i]) 
        return grad
