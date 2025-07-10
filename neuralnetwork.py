import numpy as np
from functions import *
from initdata import *
from visualizations import *

class Node:
    # init with random or np.ones params, weights are connected from prev layers
    def __init__(self, inputSize):
        self.inputSize   = inputSize
        self.inputs      = np.empty(inputSize, dtype=np.float32)
        self.weights     = np.float32(np.random.rand(inputSize) * np.sqrt(2 / inputSize))
        self.bias        = np.float32(np.random.rand(1))
        self.weightedSum = np.float32(0)

    # calculate and stores weighted sum 
    def calculateWeightedSum(self, inputs):
        self.inputs = inputs
        self.weightedSum = np.dot(self.weights, self.inputs) + self.bias
        return (self.weightedSum).item() # .item() ensures it is a scalar

    # adjust weights and bias
    def backPropogate(self, gradient):
        for i in range(self.inputSize):
            """
            print("")
            print(self.weights[i])
            """
            self.weights[i] -= LEARNINGRATE * (gradient * self.inputs[i]) # dL/dw
            """
            print("ADJUSTED BY", LEARNINGRATE * (gradient * self.inputs[i]))
            print(self.weights[i])
            print("")
            """

        self.bias -= LEARNINGRATE * gradient                              # dL/db 



# input layer will not be a layer
class Layer:
    def __init__(self, layerSize, inputSize, activationFunction):        
        self.layerSize = layerSize
        self.inputSize = inputSize                   
        self.activationFunction = activationFunction
        self.nodes = np.empty(layerSize, dtype=Node) 
        self.activations = np.empty(layerSize, dtype=np.float32)
        self.weightedSum = np.empty(layerSize, dtype=np.float32)
        for i in range(layerSize):                   
            self.nodes[i] = Node(inputSize)

    # inputs = activations from previous layer
    def calculateActivation(self, inputs):
        # calculates weighted sum for each node
        for i in range(self.layerSize):
            self.weightedSum[i] = self.nodes[i].calculateWeightedSum(inputs)

        # pass whole layer through activation function
        self.activations = self.activationFunction.forward(self.weightedSum)

        """
        print("activations: ")
        print(self.activations)
        print("")
        """
        return self.activations # returns vector of activations

    # calculates dLdz, adjusts w and b for each node in layer
    # gradient = vector of dL/da of each node 
    def backPropogate(self, gradient):
        dadz = self.activationFunction.backward(self.weightedSum, self.activations, gradient)
        dLdz = dadz * gradient

        # adjust w and b for each node
        for i in range(self.layerSize):   
            self.nodes[i].backPropogate(dLdz[i])

        return np.sum(dLdz) # return sum of dL/dz for this layer (used for backprop to prev layers)

    
class NeuralNetwork:
    def __init__(self, layerCount, trainingData, lossfunction):
        self.trainingData = trainingData                   # array of all training examples, 2d array
        self.layerCount = layerCount                       # how many layers in the network
        self.layers = np.empty(layerCount, dtype=Layer)    # empty array of layers
        self.lossfunction = lossfunction

        # these 2 arrays change after every epoch
        self.outputs = np.empty((TRAININGSIZE, CLASSCOUNT), dtype=np.float32)   # array of outputs
        self.loss = np.empty(TRAININGSIZE, dtype=np.float32)                    # array of losses

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
        self.loss[dataIndex] = self.lossfunction.forward(activations, label)   

    # passes vector of gradients to each layer recursively
    def backPropogate(self, label, dataIndex): 
        dLda = self.lossfunction.backward(self.outputs[dataIndex], label) # how loss changes w.r.t final layer output

        # calculate vector of dLda for every layer, sends this vector to layer class to calculate gradients of weights and bias for this layer
        for i in range(self.layerCount - 1, 0, -1):                               # starting from 2nd last layer,
            dLdz = self.layers[i].backPropogate(dLda)                             # sends dL/da to layer i, returns sum of dL/dz of layer i
            dLdz = np.full(len(self.layers[i - 1].nodes), dLdz, dtype=np.float32) # array of sum of dLdz (all values same)

            # note: weights + bias of layer have been adjusted by this point, the rest is calculating derivatives needed for chain rule
            dzda = np.zeros(len(self.layers[i - 1].nodes), dtype=np.float32)    # how weighted sums changes w.r.t each activation in layer[i-1]
            for j in range(len(self.layers[i - 1].nodes)):                      
                for k in range(len(self.layers[i].nodes)):                      # for each node in layer[i-1], sum the weights for each 
                    dzda[j] += self.layers[i].nodes[k].weights[j]               # node in layer ahead that connects to this node
                                                          
            dLda = dLdz * dzda


TRAININGSIZE = 200 
TESTINGSIZE = 0
CLASSCOUNT = 1  # size of label vector
LEARNINGRATE = 0.01
EPOCHS = 10000

# testing
if __name__ == "__main__":
    
    def f(x):
        return np.sin(x)

    data = np.random.uniform(-2 * np.pi, 2 * np.pi, 200)
    labels = f(data)
    data = data.reshape(-1,1)
    labels = labels.reshape(-1,1)

    trainingData = np.array(data)

    labels = np.array(labels)

    # REMEMBER: CHANGE LAYER INDEX AND LAYER COUNT
    nn = NeuralNetwork(layerCount=3, trainingData=trainingData, lossfunction=MSE)
    nn.addLayer(layerIndex=0, inputSize=1, layerSize=10, activationFunction=Tanh)
    nn.addLayer(layerIndex=1, inputSize=10, layerSize=10, activationFunction=Tanh)
    nn.addLayer(layerIndex=2, inputSize=10, layerSize=CLASSCOUNT, activationFunction=Tanh)


    losses = np.empty((EPOCHS, TRAININGSIZE), dtype=np.float32)
    for i in range(EPOCHS):
        for j in range(TRAININGSIZE):
            nn.feedForward(labels[j], j)
            print("loss: ")
            print(np.mean(nn.loss))
            print("")
            nn.backPropogate(labels[j], j)

        # 2d array of losses for each epoch
        losses[i] = nn.loss

    plt.scatter(data,labels)
    plt.scatter(data, nn.outputs, color="r")
    plt.show()

    plotLoss(EPOCHS, losses)
        