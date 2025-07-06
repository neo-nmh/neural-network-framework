import numpy as np
from functions import *
from initdata import *

TRAININGSIZE = 1
TESTINGSIZE = 0
CLASSCOUNT = 10
LEARNINGRATE = 0.1

class Node:
    # init with random params, weights are weighted connected from prev layers
    def __init__(self, inputSize):
        self.inputSize = inputSize
        self.inputs = np.empty(inputSize, dtype=np.float32)
        self.weights = np.float32(np.random.rand(inputSize))
        self.bias = np.float32(np.random.rand())

    # calculate and stores activation and weighted sum
    def calculateActivation(self, inputs, activationFunction):
        self.inputs = inputs
        z = np.dot(self.weights, self.inputs) + self.bias
        self.weightedSum = z 
        self.activation = activationFunction.forward(z)

    # adjust weights and bias
    def backPropogate(self, gradient):
        for i in range(self.inputSize):
            self.weights[i] -= LEARNINGRATE * (gradient * self.inputs[i]) # dL/dw

        self.bias -= LEARNINGRATE * gradient                              # dL/db 



# input layer will not be a layer
class Layer:
    def __init__(self, layerSize, inputSize, activationFunction):        
        self.layerSize = layerSize
        self.inputSize = inputSize                        # number of weights connected to each node
        self.activationFunction = activationFunction
        self.nodes = np.empty(self.layerSize, dtype=Node) # layers store array of nodes
        for i in range(self.layerSize):                   # init random weights for each node
            self.nodes[i] = Node(self.inputSize)

    # inputs = activations from previous layer
    def calculateActivation(self, inputs):
        activations = np.empty(self.layerSize, dtype=np.float32)
        for i in range(self.layerSize):
            self.nodes[i].calculateActivation(inputs, self.activationFunction)
            activations[i] = self.nodes[i].activation

        return activations # returns array of activations

    # for each node, adjust weights and bias
    # gradient = dL/da of each node 
    def backPropogate(self, gradient):
        dLdz = 0
        for i in range(self.layerSize):
            temp = gradient[i] * activationFunction.backward(self.nodes[i].weightedSum) # gradient * da/dz of each node
            self.nodes[i].backPropogate(temp)
            dLdz += temp

        return dLdz # return sum of dL/dz for this layer

    
class NeuralNetwork:
    def __init__(self, layerCount, trainingData):
        self.trainingData = trainingData                   # array of all training examples, 2d array
        self.layerCount = layerCount
        self.layers = np.empty(layerCount, dtype=Layer)    # empty array of layers

        # these 2 arrays change after every epoch
        self.outputs = np.empty((TRAININGSIZE,CLASSCOUNT), dtype=np.float32)              # array of outputs
        self.loss = np.empty(TRAININGSIZE, dtype=np.float32)                 # array of losses

    # inputSize will just be the previous layer's layerSize (remember when writing main function)
    def addLayer(self, layerIndex, inputSize, layerSize, activationFunction):
        self.layers[layerIndex] = Layer(layerSize, inputSize, activationFunction)

    # using 1 training example at a time
    def feedForward(self, label, dataIndex):
        activations = self.layers[0].calculateActivation(self.trainingData[dataIndex])

        for i in range(1, self.layerCount): # starting from 2nd layer
            activations = self.layers[i].calculateActivation(activations)

        # storing activations 
        self.outputs[dataIndex] = activations

        # calculate and store loss
        self.loss[dataIndex] = calculateLoss(activations, label)   

    # passes array of gradients to layer recursively
    def backPropogate(self, dataIndex): 
        print(self.trainingData[dataIndex])
        print(self.outputs[dataIndex])
        dLda = (-1 / CLASSCOUNT) * (self.trainingData[dataIndex] - self.outputs[dataIndex]) # dL/da

        for i in range(self.layerCount - 1, -1, -1):
            dLda = np.full(len(self.layers[i].nodes), dLda)    # dL/da of layer i
            temp = self.layers[i].backPropogate(dLda)          # backpropogate dL/da, returns sum of dL/dz
            dLdz = np.full(temp, len(self.layer[i].nodes))     # array of sum of dLdz (all values same)

            # array length of layer i-1 
            dzda = np.zeros(len(self.layers[i - 1].nodes), dtype=np.float32) # how weighted sums changes wrt each activation in layer[i-1]
            for j in range(len(self.layers[i - 1].nodes)):                   # for each node in layer[i-1], sum the weights going forward 
                for k in range(len(self.layers[i].nodes)):
                    dzda[j] += self.layers[i].nodes.weights[j]

            dLda = dLdz * dzda


if __name__ == "__main__":
    nn = NeuralNetwork(4, np.array([[1,2,3,4,5]]))
    nn.addLayer(layerIndex=0, inputSize=5, layerSize=10, activationFunction=ReLu)
    nn.addLayer(layerIndex=1, inputSize=10, layerSize=10, activationFunction=ReLu)
    nn.addLayer(layerIndex=2, inputSize=10, layerSize=10, activationFunction=ReLu)
    nn.addLayer(layerIndex=3, inputSize=10, layerSize=10, activationFunction=ReLu)
    nn.feedForward(label=[0,0,0,0,0,0,0,0,0,1], dataIndex=0)
    nn.backPropogate(dataIndex=0)