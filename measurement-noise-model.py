import matplotlib.pyplot as plt
import numpy as np
import scipy

t_air = 22
t_0 = 15
lam = 3
stdev = 1

def temp_model(t):
    t = t_air + (t_0 - t_air)*np.exp(-t/lam)

    return t

def get_density(x, y):
    p = 1/(stdev * np.sqrt(2 * np.pi)) * np.exp(-1/2 * ((y-temp_model(x))/stdev)**2)

    return p

# https://matplotlib.org/stable/gallery/mplot3d/surface3d.html#sphx-glr-gallery-mplot3d-surface3d-py

from matplotlib import cm
from matplotlib.ticker import LinearLocator

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

t = np.arange(0, 5, 0.1)
T = np.arange(0, 1, 0.1)
t, T= np.meshgrid(t, T)
# p = get_density(t, T)

p = t.copy()

for i in range(len(t)):
    for j in range(len(t[i])):
        p[i][j] = get_density(t[i][0], T[i][0])

surf = ax.plot_surface(t, T, p, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(0, 0.5)

# Add a color bar which maps values to colors.

plt.show()