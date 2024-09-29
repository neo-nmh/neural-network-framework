import numpy as np
from activationfunctions import *
from initdata import *

TRAININGSIZE = 0
TESTINGSIZE = 0

class Node:
    # init with random weights and bias
    # inputSize is n of weights connected from previous layer
    def __init__(self, inputSize):
        self.weights = np.random.rand(inputSize)
        self.bias = np.random.rand()
        self.activation = 0

    # calculates dot product and applies activation function
    def feedForward(self, inputs, activationFunction):
        z = np.dot(self.weights, inputs)
        self.activation = activationFunction.forward(z)

# input layer will NOT be a layer
class Layer:
    def __init__(self, layerSize, inputSize, activationFunction):        
        self.layerSize = layerSize
        self.inputSize = inputSize            # number of weights
        self.nodes = np.empty(self.layerSize) # empty array of nodes, stores activations

        # init nodes with random weights
        for i in range(self.layerSize):
            self.nodes[i] = Node(self.inputSize)

    # inputs = activations from previous layer
    # returns the activations
    def feedForward(self, inputs):
        for i in range(self.layerSize):
            self.nodes[i] = self.nodes[i].feedForward[inputs]

        return self.nodes
                                  

class NeuralNetwork:
    def __init__(self, layerCount):
        self.layerCount = layerCount
        self.layers = np.empty(layerCount)    # empty array of layers
        self.outputs = np.empty(TRAININGSIZE) # array for storing outputs

    # inputSize will just be the previous layer's layerSize (remember when writing main function)
    def addLayer(self, layerIndex, layerSize, inputSize, activationFunction):
        self.layers[layerIndex] = Layer(layerSize, inputSize, activationFunction)

    # data = entire training set
    # feeds data[n] to first layer, then feeds prev layer activations
    # store outputs to self.outputs
    # repeat for all training set
    def feedForward(self, data):
        for i in range(data):
            activations = self.layers[0].feedForward(data[i])
            for j in range(1, self.layerCount):
                activations = self.layers[j].feedForward(activations)

            outputs[i] = activations


            



if __name__ == "__main__":
    nn = NeuralNetwork(4)