import numpy as np
from activationfunctions import *
from initdata import *

TRAININGSIZE = 0
TESTINGSIZE = 0

class Node:
    # init with random weights and bias
    # inputSize is number of weights connected from previous layer
    def __init__(self, inputSize):
        self.weights = np.random.rand(inputSize)
        self.bias = np.random.rand()
        self.weightedSum = 0
        self.activation = 0

    # calculates dot product and applies activation function
    def feedForward(self, inputs, activationFunction):
        z = np.dot(self.weights, inputs) + self.bias
        self.weightedSum = z 
        self.activation = activationFunction.forward(z)


# input layer will NOT be a layer
class Layer:
    def __init__(self, layerSize, inputSize, activationFunction):        
        self.layerSize = layerSize
        self.inputSize = inputSize                        # number of weights for each node
        self.nodes = np.empty(self.layerSize, dtype=Node) # empty array of nodes, stores activations

        for i in range(self.layerSize): # init random weights for each node
            self.nodes[i] = Node(self.inputSize)

    # inputs = activations from previous layer
    def feedForward(self, inputs):
        for i in range(self.layerSize):
            self.nodes[i] = self.nodes[i].feedForward(inputs, activationFunction)

        return self.nodes # returns array of activations
                                  

class NeuralNetwork:
    def __init__(self, layerCount):
        self.layerCount = layerCount
        self.layers = np.empty(layerCount, dtype=Layer)    # empty array of layers
        self.outputs = np.empty(TRAININGSIZE) # array for storing outputs

    # inputSize will just be the previous layer's layerSize (remember when writing main function)
    def addLayer(self, layerIndex, layerSize, inputSize, activationFunction):
        self.layers[layerIndex] = Layer(layerSize, inputSize, activationFunction)

    # data = 1 training example
    # feeds data to first layer, then feeds activations forwards
    # store outputs to self.outputs
    def feedForward(self, data, index):
        activations = self.layers[0].feedForward(data)

        for i in range(1, self.layerCount): # from 2nd layer
            activations = self.layers[i].feedForward(activations)

        # index is the index of training example
        outputs[index] = activations

    def backPropogate(self):
        return 



            

if __name__ == "__main__":
    nn = NeuralNetwork(4)
    nn.addLayer(0, 5, 10, ReLu)
    print(nn.layerCount)