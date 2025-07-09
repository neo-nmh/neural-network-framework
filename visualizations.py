import numpy as np
import matplotlib.pyplot as plt

# visualize loss over epochs
def plotLoss(epochs, loss):
    averageLoss = np.empty(epochs)

    for i in range(len(loss)):
        averageLoss[i] = np.mean(loss[i])

    x = np.arange(0, epochs, 1)
    plt.plot(x, averageLoss, color="red")
    plt.ylim(0, 10)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.show()

    
