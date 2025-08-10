import numpy as np
from hyperparameters import BATCHSIZE, CLASSSIZE, LEARNINGRATE

class FullyConnectedLayer:
    def __init__(self, layerSize, inputSize, weightInitialization, activationFunction):        
        self.layerSize          = layerSize
        self.inputSize          = inputSize                   
        self.activationFunction = activationFunction
        self.weights            = weightInitialization(fanOut=layerSize, fanIn=inputSize)
        self.biases             = np.zeros(layerSize, dtype=np.float32)
        self.activations        = np.empty((BATCHSIZE, layerSize), dtype=np.float32)
        self.inputs             = np.empty((BATCHSIZE, inputSize), dtype=np.float32)
        self.weightedSum        = np.empty((BATCHSIZE, layerSize), dtype=np.float32)

    # calculate outputs batch item at a time
    def feedForward(self, input, batchItemIndex):
        # if input is from conv or pooling layer
        if input.ndim > 1:
            input = input.flatten()

        self.inputs[batchItemIndex] = input
        self.weightedSum[batchItemIndex] = self.weights @ input

        activations = self.activationFunction.forward(self.weightedSum[batchItemIndex])
        self.activations[batchItemIndex] = activations

        return activations

    # backpropagate using outputs from whole batch
    def backPropagate(self, nextLayerBatchGradients, nextLayerWeights):
        # calculate gradients to pass back 
        batchGradients = np.zeros((BATCHSIZE, self.layerSize), dtype=np.float32)
        for i in range(BATCHSIZE):
            dadz = self.activationFunction.backward(self.weightedSum[i], self.activations[i])
            batchGradients[i] = (nextLayerWeights.T @ nextLayerBatchGradients[i]) * dadz

        # change w and b
        dLdw = (batchGradients.T @ self.inputs) / BATCHSIZE
        averageGradient = batchGradients.mean(axis=0) 
        self.biases -= LEARNINGRATE * averageGradient
        self.weights -= LEARNINGRATE * dLdw

        # pass gradients to prev layer
        return batchGradients

# gradient calculation for this layer is different
# the rest is same for as FullyConnectedLayer
class OutputLayer(FullyConnectedLayer):
    def backPropagate(self, dLda):
        batchGradients = np.zeros((BATCHSIZE, CLASSSIZE), dtype=np.float32)
        for i in range(BATCHSIZE):
            dadz = self.activationFunction.backward(self.weightedSum[i], self.activations[i])
            batchGradients[i] = dLda * dadz

        dLdw = (batchGradients.T @ self.inputs) / BATCHSIZE
        averageGradient = batchGradients.mean(axis=0) 
        self.biases -= LEARNINGRATE * averageGradient
        self.weights -= LEARNINGRATE * dLdw

        return batchGradients
