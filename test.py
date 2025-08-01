import numpy as np
from functions import *
from weightinitializations import *
from initdata import *
from visualizations import *
from initdata import *

BATCHSIZE = 1
# numpy array print settings
np.set_printoptions(suppress=True, precision=2)

class ConvolutionalLayer:
    def __init__(
            self, inputSize, inputDepth, kernelSize, 
            kernelCount, stride, weightInitialization, activationFunction):
        self.inputSize          = inputSize
        self.inputDepth         = inputDepth
        self.kernelSize         = kernelSize
        self.kernelCount        = kernelCount
        self.fanIn              = (kernelSize ** 2) * inputDepth
        self.stride             = stride
        self.activationFunction = activationFunction
        self.convolutionLength  = inputSize - kernelSize + 1
        self.activationSize     = ((inputSize - kernelSize) // stride) + 1
        self.activations        = np.empty((BATCHSIZE, kernelCount, self.activationSize, self.activationSize), dtype=np.float32)
        self.inputs             = np.empty((BATCHSIZE, inputDepth, inputSize, inputSize), dtype=np.float32)
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
        self.inputs[batchItemIndex] = input

        i = 0
        patchMatrix = np.empty((self.fanIn, self.activationSize ** 2), dtype=np.float32)
        for j in range(0, self.convolutionLength, self.stride):
            for k in range(0, self.convolutionLength, self.stride):
                patchMatrix[:, i] = input[:, j:(j + self.kernelSize), k:(k + self.kernelSize)].flatten()
                i += 1

        activations = self.activationFunction.forward((self.kernelMatrix @ patchMatrix) + self.biases)
        activationsReshaped = activations.reshape(self.kernelCount, self.activationSize, self.activationSize)
        self.activations[batchItemIndex] = activationsReshaped

        # returns activations shaped as images
        return activationsReshaped

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
        kernelSize=5,
        kernelCount=10, 
        stride=1, 
        weightInitialization=heNormalConv, 
        activationFunction=ReLu)

    inputImage = trainImages[np.random.randint(1, 100)].reshape(1,28,28)
    output = convLayer.feedForward(input=inputImage, batchItemIndex=0)

    plotImage(image=inputImage.reshape(28,28), title="input")

    for i in range(len(output)):
        plotImage(image=output[i], title=f"kernel {i}")

    