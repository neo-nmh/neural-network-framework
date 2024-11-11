import numpy as np
import math
from neuralnetwork import CLASSCOUNT

# activation functions operate on whole layer at a time

class ReLu:
    def forward(activations):
        outputs = np.empty(len(activations), dtype=np.float32)
        for i in range(len(activations)):
            outputs[i] = max(0, activations[i])

        return outputs

    # ReLu doesn't use 'activations' parameter
    def backward(weightedSum, activations, gradient):
        outputs = np.empty(len(weightedSum), dtype=np.float32)
        for i in range(len(weightedSum)):
            if weightedSum[i] > 0:
                outputs[i] = 1
            else:
                outputs[i] = 0

        return outputs


class Sigmoid:
    def forward(activations):
        outputs = np.empty(len(activations), dtype=np.float32)
        for i in range(len(activations)):
            outputs[i] = 1 / (1 + math.exp(-1 * activations[i]))

        return outputs

    def backward(weightedSum, activations, gradient):
        outputs = np.empty(len(weightedSum), dtype=np.float32)
        for i in range(len(weightedSum)):
            outputs[i] = activations[i] * (1 - activations[i])

        return outputs

class Nothing:
    def forward(activation):
        return activation

    def backward(weightedSum, activation, gradient):
        return gradient


class Softmax:
    def forward(activations):
        outputs = np.empty((len(activations)), dtype=np.float32)
        # denominator is sum of exp(activations)
        expSum = 0
        for i in activations:
            expSum += math.exp(i) 

        # divide each exp(activation) by sum
        for i in range(len(activations)):
            outputs[i] = math.exp(activations[i]) / expSum

        return outputs

    # this only works with cross entropy 
    def backward(weightedSum, activations, gradient):
        return  gradient


class MSE:
    # 1/2 MSE loss function
    def forward(activations, label):
        loss = 0
        for i in range(len(activations)):
            loss += (np.square(label[i] - activations[i]))
        loss /= (2 * len(activations))

        return loss

    # dL/da = -1/n(y-a)
    def backward(activations, label):
        grad = np.empty(CLASSCOUNT, dtype=np.float32)
        for i in range(CLASSCOUNT):
            grad[i] = (-1 / CLASSCOUNT) * (label[i] - activations[i]) 

        return grad

class CrossEntropy:
    def forward(activations, label):
        activations = np.clip(activations, 1e-15, 1 - 1e-15) # prevents log(0)
        loss = np.sum(label * np.log(activations))
        return -1 * loss

    def backward(activations, label):
        return activations - label