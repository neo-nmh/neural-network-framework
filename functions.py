import numpy as np

class ReLu:
    def forward(input):
        return max(0, input)

    def backward(weightedSum):
        if weightedSum > 0:
            return 1
        else:
            return 0


# 1/2 MSE loss function
def calculateLoss(activations, label):
    loss = 0
    for i in range(len(activations)):
        loss += (np.square(label[i] - activations[i]))
        
    loss /= (2 * len(activations))

    return loss
