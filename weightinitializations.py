import numpy as np

# MLP
def heNormal(fanIn, fanOut):
    std = np.sqrt(2 / fanIn)
    return np.random.normal(loc=0, scale=std, size=(fanOut, fanIn)).astype(np.float32)
    
def heUniform(fanIn, fanOut):
    limit = np.sqrt(6 / fanIn)
    return np.random.uniform(low=-limit, high=limit, size=(fanOut, fanIn)).astype(np.float32)
    
def normal(fanIn, fanOut):
    return np.random.normal(loc=0, scale=1, size=(fanOut, fanIn)).astype(np.float32)
    
def uniform(fanIn, fanOut):
    return np.random.uniform(low=-1, high=1, size=(fanOut, fanIn)).astype(np.float32)

# Conv
def heNormalConv(fanIn, kernelSize, kernelDepth):
    std = np.sqrt(2 / fanIn)
    return np.random.normal(loc=0, scale=std, size=(kernelDepth, kernelSize, kernelSize)).astype(np.float32)