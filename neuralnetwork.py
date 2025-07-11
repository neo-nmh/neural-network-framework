import numpy as np
from functions import *
from initdata import *
from visualizations import *
from initdata import *

class Node:
    def __init__(self, inputSize):
        self.inputSize   = inputSize
        self.inputs      = np.empty(inputSize, dtype=np.float32)
        self.weights     = np.float32(np.random.rand(inputSize))
        self.bias        = np.float32(np.random.rand())
        self.weightedSum = np.float32(0)

    # calculate and stores weighted sum 
    def calculateWeightedSum(self, inputs):
        self.inputs = inputs
        self.weightedSum = np.dot(self.weights, self.inputs) + self.bias
        return self.weightedSum

    # adjust weights and bias
    def backPropogate(self, gradient):
        for i in range(self.inputSize):
            self.weights[i] -= LEARNINGRATE * (gradient * self.inputs[i]) # dL/dw
            # print("WEIGHT ADJUSTED BY", LEARNINGRATE * (gradient * self.inputs[i]))

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
        self.outputs = np.empty((TRAININGSIZE, CLASSSIZE), dtype=np.float32)   # array of outputs
        self.loss = np.empty(TRAININGSIZE, dtype=np.float32)                    # array of losses

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

    # passes vector of gradients to each layer 
    def backPropogate(self, label, dataIndex): 
        dLda = self.lossfunction.backward(self.outputs[dataIndex], label) # how loss changes w.r.t final layer output
        # print(f"loss gradient {dLda}\n")

        # calculate vector of dLda for every layer, sends this vector to layer class to calculate gradients of weights and bias for this layer
        for i in range(self.layerCount - 1, 0, -1):                               # starting from 2nd last layer,
            dLdz = self.layers[i].backPropogate(dLda)                             # sends dL/da to layer i, returns sum of dL/dz of layer i
            # print(f"layer gradient return : {dLdz}")
            dLdz = np.full(len(self.layers[i - 1].nodes), dLdz, dtype=np.float32) # array of sum of dLdz (all values same)
            # note: weights + bias of layer have been adjusted by this point, the rest is calculating derivatives needed for chain rule
            dzda = np.zeros(len(self.layers[i - 1].nodes), dtype=np.float32)    # how weighted sums changes w.r.t each activation in layer[i-1]
            for j in range(len(self.layers[i - 1].nodes)):                      
                for k in range(len(self.layers[i].nodes)):                      # for each node in layer[i-1], sum the weights for each 
                    dzda[j] += self.layers[i].nodes[k].weights[j]               # node in layer ahead that connects to this node
                                                          
            dLda = dLdz * dzda

    def classify(self, data):
        activations = self.layers[0].calculateActivation(data)

        for i in range(1, self.layerCount):
            activations = self.layers[i].calculateActivation(activations)

        return activations

# mnist_train has 60,000
# mnist_test has 10,000

TRAININGSIZE = 600 
TESTINGSIZE = 10 
FEATURESIZE = 784
CLASSSIZE = 10 
LEARNINGRATE = np.float32(0.001)
EPOCHS = 1 

if __name__ == "__main__":
    # init data
    data = initData(trainSize=TRAININGSIZE, testSize=TESTINGSIZE, classSize=CLASSSIZE)

    # init neural network
    nn = NeuralNetwork(layerCount=3, trainingData=data["trainImages"], lossfunction=CrossEntropy)
    nn.addLayer(layerIndex=0, inputSize=784, layerSize=128, activationFunction=Sigmoid)
    nn.addLayer(layerIndex=1, inputSize=128, layerSize=128, activationFunction=Sigmoid)
    nn.addLayer(layerIndex=2, inputSize=128, layerSize=CLASSSIZE, activationFunction=Softmax)

    # train and store losses
    losses = np.empty((EPOCHS * TRAININGSIZE), dtype=np.float32)
    trainingStep = 0
    for i in range(EPOCHS):
        for j in range(TRAININGSIZE):
            nn.feedForward(data["trainLabels"][j], j)
            losses[trainingStep] = nn.loss[j]
            print(f"loss: {nn.loss[j]}")
            nn.backPropogate(data["trainLabels"][j], j)
            trainingStep += 1


    # evaluate performance on test set 
    predictActivations = np.empty((TESTINGSIZE, CLASSSIZE), dtype=np.float32)
    scores = np.empty(TESTINGSIZE, dtype=np.bool)
    for i in range(TESTINGSIZE):
        predictActivations[i] = nn.classify(data["testImages"][i])
        if np.argmax(predictActivations[i]) == np.argmax(data["testLabels"][i]):
            scores[i] = True
        else:
            scores[i] = False

    print(data["testLabels"])
    print(predictActivations)
    print(scores)
    print(f"Accuracy: {np.sum(scores) / len(scores) * 100}%\n")
    plotLoss(trainingSteps=TRAININGSIZE * EPOCHS, losses=losses)