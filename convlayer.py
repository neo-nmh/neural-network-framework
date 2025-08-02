import numpy as np
from constants import BATCHSIZE

class ConvolutionalLayer:
    def __init__(
            self, inputSize, inputDepth, kernelSize, 
            kernelCount, stride, padding, weightInitialization, activationFunction):
        self.inputSize          = inputSize + (padding * 2)
        self.inputDepth         = inputDepth
        self.kernelSize         = kernelSize
        self.kernelCount        = kernelCount
        self.fanIn              = (kernelSize ** 2) * inputDepth
        self.stride             = stride
        self.padding            = padding
        self.activationFunction = activationFunction
        self.convolutionLength  = self.inputSize - kernelSize + 1
        self.activationSize     = ((self.inputSize - kernelSize) // stride) + 1
        self.activations        = np.empty((BATCHSIZE, kernelCount, self.activationSize, self.activationSize), dtype=np.float32)
        self.inputs             = np.empty((BATCHSIZE, inputDepth, self.inputSize, self.inputSize), dtype=np.float32)
        self.kernels            = np.empty((kernelCount, inputDepth, kernelSize, kernelSize), dtype=np.float32)
        self.biases             = np.zeros((kernelCount, 1), dtype=np.float32)
        self.weightedSum        = np.empty((BATCHSIZE, kernelCount, self.activationSize, self.activationSize), dtype=np.float32)
        for i in range(kernelCount):
            self.kernels[i] = weightInitialization(fanIn=(self.fanIn), kernelSize=(kernelSize), kernelDepth=(inputDepth))
            print(f"kernel {i}")
            print(self.kernels[i])
            print("")
        self.kernelMatrix = self.kernels.reshape(self.kernelCount, self.fanIn) 

    # kernelMatrix = 1 flattened kernel for each row
    # patchMatrix  = 1 flattened image patch for each column
    # activations  = 1 output image for each row
    def feedForward(self, input, batchItemIndex):
        # pad and store input
        input = np.pad(
            input, 
            pad_width=((0,0), (self.padding, self.padding), (self.padding, self.padding)),
            mode="constant",
            constant_values=0
        )
        self.inputs[batchItemIndex] = input

        # fill patch matrix
        i = 0
        patchMatrix = np.empty((self.fanIn, self.activationSize ** 2), dtype=np.float32)
        for j in range(0, self.convolutionLength, self.stride):
            for k in range(0, self.convolutionLength, self.stride):
                patchMatrix[:, i] = input[:, j:(j + self.kernelSize), k:(k + self.kernelSize)].flatten()
                i += 1

        # calculate weighted sum
        weightedSum = (self.kernelMatrix @ patchMatrix) + self.biases
        self.weightedSum[batchItemIndex] = weightedSum.reshape(self.kernelCount, self.activationSize, self.activationSize)

        # calculate activations
        activations = self.activationFunction.forward(weightedSum)
        activationsReshaped = activations.reshape(self.kernelCount, self.activationSize, self.activationSize)
        self.activations[batchItemIndex] = activationsReshaped

        # return activations shaped as images
        return activationsReshaped

    def backPropagate():
        return 0
