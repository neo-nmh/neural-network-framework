import numpy as np

class MaxPool:
    def forward(input):
        return np.max(input, axis=(1, 2), keepdims=True)

    def backward():
        pass

class AveragePool:
    def forward(input):
        return np.mean(input, axis=(1, 2), keepdims=True)

    def backward():
        pass