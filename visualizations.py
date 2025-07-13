import numpy as np
import matplotlib.pyplot as plt

# visualize loss over training steps
def plotLoss(losses):
    trainingSteps=len(losses)
    x = np.arange(0, trainingSteps, 1)
    plt.scatter(x, losses, color="red")
    plt.xlabel("training steps")
    plt.ylabel("loss")
    plt.show()

def plotImage(image, index, label):
    image = np.reshape(image, (28, 28))
    plt.imshow(image)
    plt.title(f"label: {np.argmax(label)} index: {index}")
    plt.show()

def plotBarChart(categories, values):
    plt.bar(categories, values)
    plt.show()

