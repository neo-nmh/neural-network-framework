import numpy as np

def heNormal(layerSize, inputSize):
    std = np.sqrt(2 / inputSize)
    return np.random.normal(loc=0, scale=std, size=(layerSize, inputSize)).astype(np.float32)
    
def heUniform(layerSize, inputSize):
    limit = np.sqrt(6 / inputSize)
    return np.random.uniform(low=-limit, high=limit, size=(layerSize, inputSize)).astype(np.float32)
    
def normal(layerSize, inputSize):
    return np.random.normal(loc=0, scale=1, size=(layerSize, inputSize)).astype(np.float32)
    
def uniform(layerSize, inputSize):
    return np.random.uniform(low=-1, high=1, size=(layerSize, inputSize)).astype(np.float32)
