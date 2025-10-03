import numpy as np
import scipy.integrate as integrate
from numpy import random
import matplotlib.pyplot as plt

def normal_dist(x, u, s):
    density = 1 / ( s * np.sqrt( 2 * np.pi ) ) * np.exp( -1/2 * ( ( x - u ) / s ) ** 2 )

    return density

# Given the measurements

from ohms_law import measurements

# Given p_prior

from ohms_law import p_prior

# Given p_likelihood

from ohms_law import p_likelihood

# Get p_posterior, proportional to p_prior and p_likelihood

M = 4
L = 2500
W = 1250

samples = []
for j in range(M):
    chain = []
    
    x = 5
    p = p_prior(np.array([x])) * p_likelihood(np.array([x]), measurements)
    for i in range(L):
        new_x = x + random.normal()
        
        new_p = p_prior(np.array([new_x])) * p_likelihood(np.array([new_x]), measurements)
        
        move_chance = min(1, new_p / p)
        
        if random.rand() < move_chance:
            x = new_x
            p = new_p
            
        if i > W:
            chain.append(x)

    samples.append(chain)

all_samples = []

for chain in samples:
    all_samples += chain
    
mean = np.mean(all_samples)
stddev = np.std(all_samples)

for chain in samples:
    plt.hist(chain, density = True, bins=20, alpha = 1/len(samples))

print(mean, stddev)
x = np.linspace(min(all_samples), max(all_samples), 100)
y = normal_dist(x, mean, stddev)

plt.plot(x, y, lw = 4)

plt.show()