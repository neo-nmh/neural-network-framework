import numpy as np
from functions import *
from initdata import *

TRAININGSIZE = 0
TESTINGSIZE = 0
CLASSCOUNT = 0
LEARNINGRATE = 0.1

class Node:
    # init with random params, weights are weighted connected from prev layers
    def __init__(self, inputSize):
        self.inputSize = inputSize
        self.inputs = np.empty(inputSize, dtype=np.float32)
        self.weights = np.random.rand(inputSize)
        self.bias = np.random.rand()

    # calculate and stores activation and weighted sum
    def calculateActivation(self, inputs, activationFunction):
        self.inputs = inputs
        z = np.dot(self.weights, inputs) + self.bias
        self.weightedSum = z 
        self.activation = activationFunction.forward(z)

    # adjust weights and bias
    def backPropogate(self, gradient):
        for i in range(self.inputSize):
            self.weights[i] -= LEARNINGRATE * (gradient * self.inputs[i]) # dL/dw

        self.bias -= LEARNINGRATE * gradient                              # dL/db 



# input layer will NOT be a layer
class Layer:
    def __init__(self, layerSize, inputSize, activationFunction):        
        self.layerSize = layerSize
        self.inputSize = inputSize                        # number of weights connected to each node
        self.nodes = np.empty(self.layerSize, dtype=Node) # layers store array of nodes
        for i in range(self.layerSize):                   # init random weights for each node
            self.nodes[i] = Node(self.inputSize)

    # inputs = activations from previous layer
    def calculateActivation(self, inputs):
        for i in range(self.layerSize):
            self.nodes[i] = self.nodes[i].calculateActivation(inputs, activationFunction)

        return self.nodes # returns array of activations

    # for each node, adjust weights and bias
    def backPropogate(self, gradient):
        gradArr = np.array(self.layerSize, dtype=float32)
        for i in range(self.layerSize):
            temp = gradient * activationFunction.backward(self.nodes[i].weightedSum) # da/dz of each node
            self.nodes[i].backPropogate(temp)
            gradArr[i] = temp

        return gradArr # return array of da/dz

    
class NeuralNetwork:
    def __init__(self, layerCount, trainingData):
        self.trainingData = trainingData                   # array of all training examples
        self.layerCount = layerCount
        self.layers = np.empty(layerCount, dtype=Layer)    # empty array of layers

        # these 2 arrays change after every epoch
        self.outputs = np.empty(TRAININGSIZE, dtype=np.float32)              # array of outputs
        self.loss = np.empty(TRAININGSIZE, dtype=np.float32)                 # array of losses

    # inputSize will just be the previous layer's layerSize (remember when writing main function)
    def addLayer(self, layerIndex, layerSize, inputSize, activationFunction):
        self.layers[layerIndex] = Layer(layerSize, inputSize, activationFunction)

    # using 1 training example at a time
    def feedForward(self, data, label, dataIndex):
        activations = self.layers[0].calculateActivation(data)

        for i in range(1, self.layerCount): # starting from 2nd layer
            activations = self.layers[i].calculateActivation(activations)

        # storing activations 
        outputs[dataIndex] = activations

        # calculate and store loss
        self.loss[dataIndex] = calculateLoss(activaitons, label)   

    def backPropogate(self, label, dataIndex): 
        gradient = (-1 / CLASSCOUNT) * (self.trainingData[dataIndex] - self.outputs[dataIndex]) # dL/da

        for i in range(self.layerCount - 1, 0, -1):
            temp = self.layers[i].backPropogate(gradient)





if __name__ == "__main__":
    nn = NeuralNetwork(4)
    nn.addLayer(0, 5, 10, ReLu)
    print(nn.layerCount)