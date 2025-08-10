import numpy as np
from hyperparameters import BATCHSIZE, LEARNINGRATE

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
        # reshape input if it's flattened (e.g., from MNIST data)
        if input.ndim == 1:
            # assume square image and reshape to (depth, height, width)
            imageSize = int(np.sqrt(input.shape[0] // self.inputDepth))
            input = input.reshape(self.inputDepth, imageSize, imageSize)
        
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

        # calculate weighted sum and store reshaped as image
        weightedSum = (self.kernelMatrix @ patchMatrix) + self.biases
        self.weightedSum[batchItemIndex] = weightedSum.reshape(self.kernelCount, self.activationSize, self.activationSize)

        # calculate activations
        activations = self.activationFunction.forward(weightedSum)
        activationsReshaped = activations.reshape(self.kernelCount, self.activationSize, self.activationSize)
        self.activations[batchItemIndex] = activationsReshaped

        # return activations shaped as images
        return activationsReshaped

    def backPropagate(self, nextLayerBatchGradients, nextLayerWeights=None):
        # initialize batch gradients to return to previous layer
        batchGradients = np.zeros((BATCHSIZE, self.inputDepth, self.inputSize, self.inputSize), dtype=np.float32)
        
        # initialize gradient accumulators for weights and biases
        kernelGradientsSum = np.zeros_like(self.kernels, dtype=np.float32)
        biasGradientsSum = np.zeros_like(self.biases, dtype=np.float32)
        
        # process each batch item
        for batchItem in range(BATCHSIZE):
            # get gradient of current layer activations w.r.t weighted sum using activation function
            dadz = self.activationFunction.backward(
                self.weightedSum[batchItem], 
                self.activations[batchItem]
            )
            
            # handle gradient transformation from next layer
            incomingGradient = nextLayerBatchGradients[batchItem]
            
            if nextLayerWeights is not None:
                # gradient is coming from a fully connected layer
                # transform: gradient w.r.t FC activations -> gradient w.r.t conv activations
                # this involves multiplying by the transpose of FC weights
                convActivationsFlat = nextLayerWeights.T @ incomingGradient
                # reshape to match convolutional layer output shape
                incomingGradient = convActivationsFlat.reshape(self.kernelCount, self.activationSize, self.activationSize)
            else:
                # gradient is coming from another convolutional/pooling layer
                if incomingGradient.ndim == 1:
                    # the gradient size should match the flattened activation size
                    expectedSize = self.kernelCount * self.activationSize * self.activationSize
                    if incomingGradient.size == expectedSize:
                        # reshape from flattened to (kernelCount, activationSize, activationSize)
                        incomingGradient = incomingGradient.reshape(self.kernelCount, self.activationSize, self.activationSize)
                    else:
                        raise ValueError(f"Gradient size {incomingGradient.size} doesn't match expected size {expectedSize}")
            
            # apply chain rule: gradient w.r.t activations * gradient of activations w.r.t weighted sum
            currentGradient = incomingGradient * dadz
            
            # store gradient to pass back (need to convert back to input space)
            # this involves "transposed convolution" or "deconvolution"
            inputGradient = np.zeros((self.inputDepth, self.inputSize, self.inputSize), dtype=np.float32)
            
            # for each kernel and each position in the output
            for kernelIdx in range(self.kernelCount):
                for i in range(self.activationSize):
                    for j in range(self.activationSize):
                        # calculate input region indices
                        input_i_start = i * self.stride
                        input_i_end = input_i_start + self.kernelSize
                        input_j_start = j * self.stride  
                        input_j_end = input_j_start + self.kernelSize
                        
                        # accumulate gradient for input region
                        inputGradient[:, input_i_start:input_i_end, input_j_start:input_j_end] += (
                            currentGradient[kernelIdx, i, j] * self.kernels[kernelIdx]
                        )
            
            batchGradients[batchItem] = inputGradient
            
            # calculate gradients wrt kernel weights and biases
            # gradient of weighted sum wrt kernel weights is the input patches
            # gradient of weighted sum wrt biases is 1
            
            # for each kernel
            for kernelIdx in range(self.kernelCount):
                # Gradient w.r.t bias (sum over all spatial positions)
                biasGradientsSum[kernelIdx, 0] += np.sum(currentGradient[kernelIdx])
                
                # gradient w.r.t kernel weights
                kernelGradient = np.zeros((self.inputDepth, self.kernelSize, self.kernelSize), dtype=np.float32)
                
                # for each position in the output
                for i in range(self.activationSize):
                    for j in range(self.activationSize):
                        # get corresponding input patch
                        input_i_start = i * self.stride
                        input_i_end = input_i_start + self.kernelSize
                        input_j_start = j * self.stride
                        input_j_end = input_j_start + self.kernelSize
                        
                        inputPatch = self.inputs[batchItem, :, input_i_start:input_i_end, input_j_start:input_j_end]
                        
                        # accumulate gradient
                        kernelGradient += currentGradient[kernelIdx, i, j] * inputPatch
                
                kernelGradientsSum[kernelIdx] += kernelGradient
        
        # apply gradient descent (average gradients over batch)
        avgKernelGradients = kernelGradientsSum / BATCHSIZE
        avgBiasGradients = biasGradientsSum / BATCHSIZE
        
        # update weights and biases
        self.kernels -= LEARNINGRATE * avgKernelGradients
        self.biases -= LEARNINGRATE * avgBiasGradients
        
        # update kernel matrix (flattened version used in forward pass)
        self.kernelMatrix = self.kernels.reshape(self.kernelCount, self.fanIn)
        
        # remove padding from gradients before returning
        if self.padding > 0:
            batchGradientsUnpadded = np.zeros((BATCHSIZE, self.inputDepth, self.inputSize - 2*self.padding, self.inputSize - 2*self.padding), dtype=np.float32)
            for batchItem in range(BATCHSIZE):
                batchGradientsUnpadded[batchItem] = batchGradients[batchItem, :, self.padding:-self.padding, self.padding:-self.padding]
            return batchGradientsUnpadded
        else:
            return batchGradients
