import numpy as np
import matplotlib.pyplot as plt

# visualize loss over epochs
def plotLoss(epochs, loss):
    averageLoss = np.empty(epochs)

    for i in range(len(loss)):
        averageLoss[i] = np.mean(loss[i])

    x = np.arange(0, epochs, 1)
    plt.plot(x, averageLoss, color="red")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.show()


"""
def f(x):
    return np.sin(x)

x = np.linspace(-2 * np.pi, 2 * np.pi, 1000)
r = np.random.uniform(-2 * np.pi, 2 * np.pi, 100)
y = f(x)
print(r.shape)
z = f(r)
print(z.shape)

plt.plot(x,y)
plt.scatter(r,z, color="r")
plt.show()
"""