import matplotlib.pyplot as plt
import numpy as np
from numpy import random

def density(x, u=0, s=1):
    return 1/(s * np.sqrt(2 * np.pi)) * np.exp(-1/2 * (x-u)**2)
    # return np.cos(x)/2

x = 0

samples = [x]

for i in range(2500):
    new_x = x + random.normal(0, 1)

    d = density(x)
    new_d = density(new_x)

    p = min(1, new_d/d)

    if random.rand() < p:
        x = new_x
        samples.append(x)

plt.hist(samples, density = True)

x = np.arange(min(samples), max(samples), 0.01)
y = density(x)

plt.plot(x, y)

plt.show()