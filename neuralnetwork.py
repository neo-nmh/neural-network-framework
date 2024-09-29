import numpy as np
from activationfunctions import *
from initdata import *


class Node:
    # init with random weights and bias
    # inputCount is n of weights connected from prev
    def __init__(self, inputCount):
        self.weights = np.random.rand(inputCount)
        self.bias = np.random.rand()

    # calculates dot product and applies activation function
    def calcActivation(self, inputs, activationFunction):
        z = np.dot(self.weights, inputs)
        self.activation = activationFunction.forward(z)

class Layer:
    # for the first layer, inputs will be training data
    # for the rest, inputs will be activations from prev layers
    def __init__(self, layerSize, inputs, activationFunction):        

        self.size = layerSize
        self.inputCount = len(inputs)

        # init nodes
        # remove
        self.nodes = np.empty(self.size)
        for i in range(self.size):
            self.nodes[i] = Node(self.inputCount).calcActivation(inputs, activationFunction) 
                                                    # remove

class NeuralNetwork:
    def __init__(layerCount):
        self.layers = np.empty(layerCount)

    def addLayer(self, layerIndex, layerSize, activationFunction):
        self.layers[layerIndex] = Layer(layerSize, activationFunction)



x = Layer(10, [1, 2, 3], ReLu)
print(x.nodes)