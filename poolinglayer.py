import numpy as np
from constants import BATCHSIZE

class PoolingLayer:
    def __init__(self, inputSize, inputDepth, kernelSize, stride, poolingFunction):
        self.inputSize         = inputSize
        self.inputDepth        = inputDepth
        self.kernelSize        = kernelSize
        self.stride            = stride
        self.poolingFunction   = poolingFunction
        self.convolutionLength = inputSize - kernelSize + 1
        self.activationSize    = ((inputSize - kernelSize) // stride) + 1
        self.activations       = np.empty((BATCHSIZE, inputDepth, self.activationSize, self.activationSize), dtype=np.float32)
        self.inputs            = np.empty((BATCHSIZE, inputDepth, self.inputSize, self.inputSize), dtype=np.float32)

    def feedForward(self, input, batchItemIndex):
        self.inputs[batchItemIndex] = input

        # calculate activations
        row = 0
        for j in range(0, self.convolutionLength, self.stride):
            col = 0
            for k in range(0, self.convolutionLength, self.stride):
                self.activations[batchItemIndex, :, row, col] = (
                    self.poolingFunction.forward(
                            input[:, j:(j + self.kernelSize), k:(k + self.kernelSize)]
                    ).reshape(-1)
                )
                col += 1
            row += 1

        # return activations shaped as images
        return self.activations[batchItemIndex]

    def backPropagate(self, nextLayerBatchGradients):
        return 0
