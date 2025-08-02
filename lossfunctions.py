import numpy as np
from constants import CLASSSIZE

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
