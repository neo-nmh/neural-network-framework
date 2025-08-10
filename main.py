import numpy as np
from neuralnetwork import *
from activationfunctions import *
from lossfunctions import *
from weightinitializations import *
from initdata import *
from visualizations import *
from hyperparameters import *

# numpy array print settings
np.set_printoptions(suppress=True, precision=2)

if __name__ == "__main__":
    # init data
    data = initData(trainSize=TRAININGSIZE, testSize=TESTINGSIZE, classSize=CLASSSIZE)
    trainImages = data["trainImages"]
    testImages = data["testImages"]
    trainLabels = data["trainLabels"]
    testLabels = data["testLabels"]

    # init network
    nn = NeuralNetwork(layerCount=3, lossfunction=CrossEntropy)
    nn.addLayer(
        layerType="convolutional", 
        layerIndex=0,
        inputSize=28, 
        inputDepth=1, 
        kernelSize=5, 
        kernelCount=3, 
        stride=1,
        padding=2,
        weightInitialization=heNormalConv,
        activationFunction=ReLu
        )
    nn.addLayer(
        layerType="fullyConnected", 
        layerIndex=1, 
        inputSize=2352, 
        layerSize=128, 
        weightInitialization=heNormal, 
        activationFunction=ReLu
        )
    nn.addLayer(
        layerType="output", 
        layerIndex=2, 
        inputSize=128, 
        layerSize=CLASSSIZE, 
        weightInitialization=heNormal, 
        activationFunction=Softmax
        )

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
    print(f"gradient steps: {len(losses)}")
    print(f"batch size:     {BATCHSIZE}")
    print(f"learning rate:  {LEARNINGRATE:.2g}")
    print(f"EPOCHS:         {EPOCHS}")
    print(f"correct:        {correct}/{TESTINGSIZE}")
    print(f"accuracy:      {round((correct / TESTINGSIZE) * 100, 1)}%")

    # plot loss
    plotLoss(losses)
