import numpy as np
import pandas as pd
from visualizations import *

def initData(trainSize, testSize, classSize):
    trainData = pd.read_csv('./data/mnist_train.csv', nrows=trainSize).to_numpy()
    trainLabels = trainData[:, 0]
    trainImages = np.delete(trainData, 0, 1) / 255

    testData = pd.read_csv('./data/mnist_test.csv', nrows=testSize).to_numpy()
    testLabels = testData[:, 0]
    testImages = np.delete(testData, 0, 1) / 255

    # convert labels to label vector
    trainLabels = np.eye(classSize)[trainLabels]
    testLabels = np.eye(classSize)[testLabels]

    data = {
        "trainLabels": np.float32(trainLabels),
        "trainImages": np.float32(trainImages),
        "testLabels": np.float32(testLabels),
        "testImages": np.float32(testImages),
    }

    return data 

