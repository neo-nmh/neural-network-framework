import numpy as np
from functions import *
from weightinitializations import *
from poolingfunctions import *
from initdata import *
from visualizations import *
from initdata import *

BATCHSIZE = 1
# numpy array print settings
np.set_printoptions(suppress=True, precision=2)

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

        # calculate activations
        activations = self.activationFunction.forward((self.kernelMatrix @ patchMatrix) + self.biases)
        activationsReshaped = activations.reshape(self.kernelCount, self.activationSize, self.activationSize)
        self.activations[batchItemIndex] = activationsReshaped

        # return activations shaped as images
        return activationsReshaped

    def backPropagate():
        return 0

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

    def backPropagate():
        return 1

TRAININGSIZE = 100
TESTINGSIZE = 1 
FEATURESIZE = 28
BATCHSIZE = 1
CLASSSIZE = 10 
LEARNINGRATE = np.float32(0.001)
EPOCHS = 1

if __name__ == "__main__":
    # init data
    data = initData(trainSize=TRAININGSIZE, testSize=TESTINGSIZE, classSize=CLASSSIZE)
    trainImages = data["trainImages"]
    testImages = data["testImages"]
    trainLabels = data["trainLabels"]
    testLabels = data["testLabels"]

    convLayer = ConvolutionalLayer(
        inputSize=28, 
        inputDepth=1,
        kernelSize=3,
        kernelCount=5, 
        stride=1, 
        padding=1,
        weightInitialization=heNormalConv, 
        activationFunction=ReLu
    )

    poolingLayer = PoolingLayer(
        inputSize=convLayer.activationSize,
        inputDepth=convLayer.kernelCount,
        kernelSize=2,
        stride=2,
        poolingFunction=MaxPool
    )

    inputImage = trainImages[np.random.randint(1, 100)].reshape(1,28,28)
    convOutput = convLayer.feedForward(input=inputImage, batchItemIndex=0)
    poolingOutput = poolingLayer.feedForward(input=convOutput, batchItemIndex=0)

    plotImage(image=convLayer.inputs[0, 0], title="input")

    print(f"input shape: {convLayer.inputSize}")
    print(f"conv shape : {convOutput.shape}")
    for i in range(len(convOutput)):
        plotImage(image=convOutput[i], title=f"kernel {i}")

    print(f"pool shape : {poolingOutput.shape}")
    for i in range(len(poolingOutput)):
        plotImage(image=poolingOutput[i], title=f"kernel {i}")

    