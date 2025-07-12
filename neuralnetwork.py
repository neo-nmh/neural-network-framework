import numpy as np
from functions import *
from initdata import *
from visualizations import *
from initdata import *

class Node:
    def __init__(self, inputSize):
        self.inputSize   = inputSize
        self.inputs      = np.empty(inputSize, dtype=np.float32)
        self.weights     = np.float32(np.random.uniform(-1.0 / np.sqrt(inputSize), 1.0 / np.sqrt(inputSize), inputSize))
        self.bias        = np.float32(0)
        self.weightedSum = np.float32(0)

    # calculate and stores weighted sum 
    def calculateWeightedSum(self, inputs):
        self.inputs = inputs
        self.weightedSum = np.dot(self.weights, self.inputs) + self.bias
        return self.weightedSum

    # adjust weights and bias
    def backPropagate(self, gradient):
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
    def backPropagate(self, gradient):
        dadz = self.activationFunction.backward(self.weightedSum, self.activations, gradient)
        dLdz = dadz * gradient

        # adjust w and b for each node
        for i in range(self.layerSize):   
            self.nodes[i].backPropagate(dLdz[i])

        # Calculate gradient for each input to this layer
        input_gradients = np.zeros(self.inputSize, dtype=np.float32)
        for i in range(self.inputSize):      # for each input
            for j in range(self.layerSize):  # sum over all nodes in this layer
                input_gradients[i] += dLdz[j] * self.nodes[j].weights[i]
        
        return input_gradients

    
class NeuralNetwork:
    def __init__(self, layerCount, trainingData, lossfunction):
        self.trainingData = trainingData                   # array of all training examples, 2d array
        self.layerCount = layerCount                       # how many layers in the network
        self.layers = np.empty(layerCount, dtype=Layer)    # empty array of layers
        self.lossfunction = lossfunction

        # these 2 arrays change after every epoch
        self.outputs = np.empty((TRAININGSIZE, CLASSSIZE), dtype=np.float32)    # array of outputs
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

    def backPropagate(self, label, dataIndex):
        dLda = self.lossfunction.backward(self.outputs[dataIndex], label)

        for i in range(self.layerCount - 1, -1, -1):
            dLda = self.layers[i].backPropagate(dLda) 


    def classify(self, data):
        activations = self.layers[0].calculateActivation(data)

        for i in range(1, self.layerCount):
            activations = self.layers[i].calculateActivation(activations)

        return activations

# mnist_train has 60,000
# mnist_test has 10,000

TRAININGSIZE = 60000
TESTINGSIZE = 2000 
FEATURESIZE = 784
CLASSSIZE = 10 
LEARNINGRATE = np.float32(0.001)
EPOCHS = 1 

if __name__ == "__main__":
    # init data
    data = initData(trainSize=TRAININGSIZE, testSize=TESTINGSIZE, classSize=CLASSSIZE)

    # init neural network
    nn = NeuralNetwork(layerCount=3, trainingData=data["trainImages"], lossfunction=MSE)
    nn.addLayer(layerIndex=0, inputSize=784, layerSize=256, activationFunction=ReLu)
    nn.addLayer(layerIndex=1, inputSize=256, layerSize=128, activationFunction=ReLu)
    nn.addLayer(layerIndex=2, inputSize=128, layerSize=CLASSSIZE, activationFunction=ReLu)

    # train and store losses
    losses = np.empty((EPOCHS * TRAININGSIZE), dtype=np.float32)
    trainingStep = 0
    for i in range(EPOCHS):
        for j in range(TRAININGSIZE):
            nn.feedForward(data["trainLabels"][j], j)
            losses[trainingStep] = nn.loss[j]
            print(f"loss: {nn.loss[j]}")
            nn.backPropagate(data["trainLabels"][j], j)
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