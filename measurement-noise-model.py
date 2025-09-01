# https://matplotlib.org/stable/gallery/mplot3d/surface3d.html#sphx-glr-gallery-mplot3d-surface3d-py

import matplotlib.pyplot as plt
import numpy as np
import scipy

t_air = 22
t_0 = 15
lam = 3
stdev = 3

def temp_model(t):
    t = t_air + (t_0 - t_air)*np.exp(-t/lam)

    return t

def get_density(x, y):
    p = 1/(stdev * np.sqrt(2 * np.pi)) * np.exp(-1/2 * ((y-temp_model(x))/stdev)**2)

    return p

from matplotlib import cm
from matplotlib.ticker import LinearLocator

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

t = np.arange(0, 15, 0.1)
T = np.arange(10, 25, 0.1)
t, T= np.meshgrid(t, T)

p = t.copy()

for i in range(len(t[0])):
    for j in range(len(t)):
        p[j][i] = get_density(t[0][i], T[j][0])

surf = ax.plot_surface(t, T, p, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(0, 0.5)

# Add a color bar which maps values to colors.

plt.show()