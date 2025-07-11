import numpy as np
import math
from neuralnetwork import CLASSSIZE


# activation functions operate on whole layers at a time
class ReLu:
    def forward(activations):
        return (activations > 0) * activations

    # ReLu doesn't use 'activations' parameter
    def backward(weightedSum, activations, gradient):
        return weightedSum > 0

class Sigmoid:
    def forward(activations):
        return 1 / (1 + np.exp(-1 * activations))

    def backward(weightedSum, activations, gradient):
        return activations * (1 - activations)

class Tanh:
    def forward(activations):
        return np.tanh(activations)

    def backward(weightedSum, activations, gradient):
        return 1 - (activations ** 2)


class Nothing:
    def forward(activation):
        return activation

    def backward(weightedSum, activation, gradient):
        return gradient


class Softmax:
    def forward(activations):
        # denominator is sum of exp(activations)
        expSum = np.sum(np.exp(activations))
        return np.exp(activations) / expSum

    # this only works with cross entropy 
    def backward(weightedSum, activations, gradient):
        return gradient


# 1/2 MSE 
class MSE:
    def forward(activations, label):
        loss = np.sum((np.square(label - activations)))
        return loss / (2 * len(activations))

    # dL/da = -1/n(y-a)
    def backward(activations, label):
        return (-1 / CLASSSIZE) * (label - activations)

class CrossEntropy:
    def forward(activations, label):
        # activations = np.clip(activations, 1e-15, 1 - 1e-15) # prevents log(0)
        return -1 * np.sum(label * np.log(activations))

    def backward(activations, label):
        return activations - label