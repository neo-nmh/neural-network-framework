import numpy as np
from fullyconnectedlayer import FullyConnectedLayer, OutputLayer
from convlayer import ConvolutionalLayer
from poolinglayer import PoolingLayer
from constants import TRAININGSIZE, BATCHSIZE, CLASSSIZE

# numpy array print settings
np.set_printoptions(suppress=True, precision=2)

class NeuralNetwork:
    def __init__(self, layerCount, lossfunction):
        self.layerCount   = layerCount                       
        self.layers       = [0] * layerCount
        self.lossfunction = lossfunction
        # these 2 arrays change after every epoch
        self.activations  = np.empty((TRAININGSIZE // BATCHSIZE, BATCHSIZE, CLASSSIZE), dtype=np.float32)   
        self.loss         = np.empty((TRAININGSIZE // BATCHSIZE, BATCHSIZE), dtype=np.float32)                   

    def addLayer(self, layerIndex, **kwargs):
        layerType = kwargs.get('layerType')
        
        if layerType == "fullyConnected":
            self.layers[layerIndex] = FullyConnectedLayer(
                kwargs['layerSize'], kwargs['inputSize'], 
                kwargs['weightInitialization'], kwargs['activationFunction']
            )
        elif layerType == "output":
            self.layers[layerIndex] = OutputLayer(
                kwargs['layerSize'], kwargs['inputSize'], 
                kwargs['weightInitialization'], kwargs['activationFunction']
            )
        elif layerType == "convolutional":
            self.layers[layerIndex] = ConvolutionalLayer(
                kwargs['inputSize'], kwargs['inputDepth'], kwargs['kernelSize'],
                kwargs['kernelCount'], kwargs['stride'], kwargs['padding'],
                kwargs['weightInitialization'], kwargs['activationFunction']
            )
        elif layerType == "pooling":
            self.layers[layerIndex] = PoolingLayer(
                kwargs['inputSize'], kwargs['inputDepth'], kwargs['kernelSize'],
                kwargs['stride'], kwargs['poolingFunction']
            )
        
    def feedForward(self, input, label, batchIndex, batchItemIndex):
        activations = self.layers[0].feedForward(input, batchItemIndex)
        for i in range(1, self.layerCount): 
            activations = self.layers[i].feedForward(activations, batchItemIndex)

        self.activations[batchIndex, batchItemIndex] = activations
        self.loss[batchIndex, batchItemIndex] = self.lossfunction.forward(activations, label)   

        return activations

    def backPropagate(self, batchLabels, batchIndex):
        dLda = np.zeros(CLASSSIZE, dtype=np.float32)
        for i in range(BATCHSIZE):
            dLda += self.lossfunction.backward(self.activations[batchIndex, i], batchLabels[i])
        dLda /= BATCHSIZE

        # calculate dLdz for whole batch and update last layer
        batchGradient = self.layers[self.layerCount - 1].backPropagate(dLda) 

        for i in range(self.layerCount - 2, -1, -1):
            if type(self.layers[i]).__name__ == "FullyConnectedLayer": 
                # fully connected layer
                batchGradient = self.layers[i].backPropagate(batchGradient, self.layers[i + 1].weights) 
            else:
                # convolutional or pooling layer
                batchGradient = self.layers[i].backPropagate(batchGradient)

        print(np.mean(self.loss[batchIndex]))
        return np.mean(self.loss[batchIndex])