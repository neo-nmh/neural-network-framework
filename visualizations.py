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

def plotImage(image, title):
    image = np.reshape(image, (len(image), len(image)))
    plt.imshow(image)
    plt.title(title)
    plt.colorbar()
    plt.show()

def plotMultipleImages(image):
    image = np.reshape(image, (len(image), len(image)))
    plt.imshow(image)
    plt.show()

def plotBarChart(categories, values):
    plt.bar(categories, values)
    plt.show()

