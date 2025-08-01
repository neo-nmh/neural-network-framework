import numpy as np
from neuralnetwork import CLASSSIZE


# activation functions operate on whole layers at a time
class ReLu:
    def forward(weightedSum):
        return np.maximum(0, weightedSum)

    # ReLu doesn't use 'activations' parameter
    def backward(weightedSum, activations):
        return (weightedSum > 0).astype(int)

class Sigmoid:
    def forward(weightedSum):
        return 1 / (1 + np.exp(-1 * weightedSum))

    def backward(weightedSum, activations):
        return activations * (1 - activations)

class Tanh:
    def forward(weightedSum):
        return np.tanh(weightedSum)

    def backward(weightedSum, activations):
        return 1 - (activations ** 2)

class Nothing:
    def forward(weightedSum):
        return weightedSum

    def backward(weightedSum, activations):
        return 1

class Softmax:
    def forward(weightedSum):
        # shift values by max to prevent overflow
        shifted = weightedSum - np.max(weightedSum)
        expSum = np.sum(np.exp(shifted))
        return np.exp(shifted) / expSum

    # this only works with cross entropy 
    def backward(weightedSum, activations):
        return 1


# loss functions
class MSE:
    def forward(activations, label):
        loss = np.sum((np.square(label - activations)))
        return loss / (2 * len(activations))

    # dL/da = 1/n(a-y)
    def backward(activations, label):
        return (1 / CLASSSIZE) * (activations - label)

class CrossEntropy:
    def forward(activations, label):
        activations = np.clip(activations, 1e-12, 1. - 1e-12) # prevents log(0)
        return -1 * np.sum(label * np.log(activations))

    # this only works with softmax 
    def backward(activations, label):
        return activations - label

# pooling functions
# class MaxPool


# class AveragePool