import numpy as np
from functions import *
from weightinitializations import *
from initdata import *
from visualizations import *
from initdata import *

np.set_printoptions(suppress=True, precision=2)

class Layer:
    def __init__(self, layerSize, inputSize, weightInitialization, activationFunction):        
        self.layerSize = layerSize
        self.inputSize = inputSize                   
        self.activationFunction = activationFunction
        self.weights = weightInitialization(layerSize=layerSize, inputSize=inputSize)
        self.biases = np.zeros(layerSize, dtype=np.float32)
        self.activations = np.empty((BATCHSIZE, layerSize), dtype=np.float32)
        self.weightedSum = np.empty((BATCHSIZE, layerSize), dtype=np.float32)

    def feedForward(self, input, batchItemIndex):
        self.input = input
        self.weightedSum[batchItemIndex] = np.matmul(self.weights, input)
        self.activations[batchItemIndex] = self.activationFunction.forward(self.weightedSum[batchItemIndex])
        return self.activations[batchItemIndex]

    def backPropagate(self, nextLayerBatchGradients, nextLayerWeights):
        # calculate batch gradient
        batchGradients = np.zeros((BATCHSIZE, self.layerSize), dtype=np.float32)
        for i in range(BATCHSIZE):
            dadz = self.activationFunction.backward(self.weightedSum[i], self.activations[i])
            batchGradients[i] = (np.matmul(np.transpose(nextLayerWeights), nextLayerBatchGradients[i])) * dadz

        # calculate average and change weights + biases
        averageGradient = np.sum(batchGradients, axis=0) / BATCHSIZE
        self.biases -= LEARNINGRATE * averageGradient
        self.weights -= LEARNINGRATE * np.outer(averageGradient, self.input)

        # return to pass to prev layers
        return batchGradients

class OutputLayer(Layer):
    def backPropagate(self, dLda):
        # calculate batch gradient
        batchGradients = np.zeros((BATCHSIZE, CLASSSIZE), dtype=np.float32)
        for i in range(BATCHSIZE):
            dadz = self.activationFunction.backward(self.weightedSum[i], self.activations[i])
            batchGradients[i] = dLda * dadz

        # calculate average and change weights + biases
        averageGradient = np.sum(batchGradients, axis=0) / BATCHSIZE
        self.biases -= LEARNINGRATE * averageGradient
        self.weights -= LEARNINGRATE * np.outer(averageGradient, self.input)

        # return to pass to prev layers
        return batchGradients

    
class NeuralNetwork:
    def __init__(self, layerCount, lossfunction):
        self.layerCount = layerCount                       
        self.layers = [0] * layerCount
        self.lossfunction = lossfunction

        # these 2 arrays change after every epoch
        self.outputs = np.empty((TRAININGSIZE // BATCHSIZE, BATCHSIZE, CLASSSIZE), dtype=np.float32)   
        self.loss = np.empty((TRAININGSIZE // BATCHSIZE, BATCHSIZE), dtype=np.float32)                   

    def addLayer(self, layerIndex, inputSize, layerSize, layerType, weightInitialization, activationFunction):
        if layerType == "hidden":
            self.layers[layerIndex] = Layer(layerSize, inputSize, weightInitialization, activationFunction)
        elif layerType == "output":
            self.layers[layerIndex] = OutputLayer(layerSize, inputSize, weightInitialization, activationFunction)
        
    def feedForward(self, input, label, batchIndex, batchItemIndex):
        activations = self.layers[0].feedForward(input, batchItemIndex)
        for i in range(1, self.layerCount): 
            activations = self.layers[i].feedForward(activations, batchItemIndex)

        self.outputs[batchIndex, batchItemIndex] = activations

        self.loss[batchIndex, batchItemIndex] = self.lossfunction.forward(activations, label)   

        return activations

    def backPropagate(self, batchLabels, batchIndex):
        dLda = np.zeros(CLASSSIZE, dtype=np.float32)
        for i in range(BATCHSIZE):
            dLda += self.lossfunction.backward(self.outputs[batchIndex, i], batchLabels[i])

        dLda /= BATCHSIZE
        batchGradient = self.layers[self.layerCount - 1].backPropagate(dLda) # returns dLdz for batch

        for i in range(self.layerCount - 2, -1, -1):
            batchGradient = self.layers[i].backPropagate(batchGradient, self.layers[i + 1].weights) 

        print(np.mean(self.loss[batchIndex]))
        return np.mean(self.loss[batchIndex])

# mnist_train has 60,000
# mnist_test has 10,000
# 28 x 28 greyscale images of numbers

TRAININGSIZE = 60000
TESTINGSIZE = 10000 
FEATURESIZE = 784
BATCHSIZE = 1
CLASSSIZE = 10 
LEARNINGRATE = np.float32(0.001)
EPOCHS = 30

if __name__ == "__main__":
    # init data
    data = initData(trainSize=TRAININGSIZE, testSize=TESTINGSIZE, classSize=CLASSSIZE)
    trainImages = data["trainImages"]
    testImages = data["testImages"]
    trainLabels = data["trainLabels"]
    testLabels = data["testLabels"]

    # init network
    nn = NeuralNetwork(layerCount=2, lossfunction=CrossEntropy)
    nn.addLayer(layerIndex=0, inputSize=784, layerSize=128, layerType="hidden", weightInitialization=heNormal, activationFunction=ReLu)
    nn.addLayer(layerIndex=1, inputSize=128, layerSize=CLASSSIZE, layerType="output", weightInitialization=heNormal, activationFunction=Softmax)

    # train network
    losses = []
    for i in range(EPOCHS):
        print(f"epoch: {i}")
        dataIndex = 0
        # shuffle training data 
        indices = np.arange(TRAININGSIZE)
        np.random.shuffle(indices)
        trainImages[:] = trainImages[indices]
        trainLabels[:] = trainLabels[indices]
        for j in range(TRAININGSIZE // BATCHSIZE):
            batchLabels = []
            for k in range(BATCHSIZE):
                nn.feedForward(input=trainImages[dataIndex], label=trainLabels[dataIndex], batchIndex=j, batchItemIndex=k)
                batchLabels.append(trainLabels[dataIndex])
                dataIndex += 1
            losses.append(nn.backPropagate(batchLabels=batchLabels, batchIndex=j))
    
    # test network
    correct = 0
    for i in range(TESTINGSIZE):
        activations = nn.feedForward(input=testImages[i], label=testLabels[i], batchIndex=0, batchItemIndex=0)
        print(f"activations: {activations}")
        print(f"label:       {testLabels[i]}")
        print("")
        if np.argmax(activations) == np.argmax(testLabels[i]):
            correct += 1
    
    print(f"training size:  {TRAININGSIZE}")
    print(f"testing size:   {TESTINGSIZE}")
    print(f"training steps: {len(losses)}")
    print(f"batch size:     {BATCHSIZE}")
    print(f"learning rate:  {LEARNINGRATE:.2g}")
    print(f"EPOCHS:         {EPOCHS}")
    print(f"correct:        {correct}/{TESTINGSIZE}")
    print(f"accuracy:      {round((correct / TESTINGSIZE) * 100, 1)}%")

    # plot loss
    plotLoss(losses)
