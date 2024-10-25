import numpy as np
import math
from neuralnetwork import CLASSCOUNT

# activation functions operate on whole layer at a time

class ReLu:
    def forward(activations):
        outputs = np.empty(len(activations), dtype=np.float32)
        for i in range(len(activations)):
            outputs[i] = max(0, activations[i])

        return outputs

    # ReLu doesn't use 'activations' parameter
    def backward(weightedSum, activations):
        outputs = np.empty(len(weightedSum), dtype=np.float32)
        for i in range(len(weightedSum)):
            if weightedSum[i] > 0:
                outputs[i] = 1
            else:
                outputs[i] = 0

        return outputs


class Sigmoid:
    def forward(activations):
        outputs = np.empty(len(activations), dtype=np.float32)
        for i in range(len(activations)):
            outputs[i] = 1 / (1 + math.exp(-1 * activations[i]))

        return outputs

    def backward(weightedSum, activations):
        outputs = np.empty(len(weightedSum), dtype=np.float32)
        for i in range(len(weightedSum)):
            outputs[i] = activations[i] * (1 - activations[i])

        return outputs

class Softmax:
    def forward(activations):
        outputs = np.empty((len(activations)), dtype=np.float32)
        # denominator is sum of exp(activations)
        expSum = 0
        for i in activations:
            expSum += math.exp(i) 

        # divide each exp(activation) by sum
        for i in range(len(activations)):
            outputs[i] = math.exp(activations[i]) / expSum

        return outputs

    # takes in a vector of weighted sums (z) with n rows
    # derivative w.r.t z[i] is a vector of n rows,
    # where nth row is how activation[n] changes w.r.t z[i]
    # these vectors can be represented as a jacobian matrix (n x n)
    def backward(weightedSum, activations):
        # temporary array for each column of jacobian 
        print("")
        print("inside softmax backward")
        print("weightedsum", weightedSum)
        print("activations", activations)
        jacobian = np.empty(len(weightedSum), dtype=np.float32)

        # sum of each column of jacobian
        outputs = np.empty(len(weightedSum), dtype=np.float32)

        # i = inputs, j = outputs
        # when i == j, derivative is different
        for i in range(len(weightedSum)):
            for j in range(len(weightedSum)):
                if i == j:
                    jacobian[j] = activations[i] * (1 - activations[i])
                else:
                    jacobian[j] = (-1 * activations[j]) * activations[i]
            outputs[i] = np.sum(jacobian)
        print("outputs ", outputs)

        print("")
        return outputs # returns sum of each column of jacobian matrix


class Loss:
    # 1/2 MSE loss function
    def forward(activations, label):
        loss = 0
        label = np.array(label)
        for i in range(len(activations)):
            loss += (np.square(label[i] - activations[i]))
        loss /= (2 * len(activations))
        return loss

    # dL/da = -1/n(y-a)
    def backward(activations, label):
        label = np.array(label)
        grad = np.empty(CLASSCOUNT, dtype=np.float32)
        for i in range(CLASSCOUNT):
            grad[i] = (-1 / CLASSCOUNT) * (label[i] - activations[i]) 
        return grad
