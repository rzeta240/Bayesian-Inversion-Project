import matplotlib.pyplot as plt
import numpy as np

from bayesian_machine import metropolis_sample_from_posterior, normal_dist

def p_prior(u):
    u = u[0]
    
    if u >= 0 and u < 100:
        return 1/100
    else:
        return 0

def p_likelihood(u, d):
    densities = []
    
    for i in range(len(d)):
        densities.append(normal_dist(d[i][0] / d[i][1], u[0], 10))
    
    return np.prod(densities)

measurements = np.array([
  [5000, 57],
  [9000, 102],
  [6000, 68],
  [10000, 114],
  [20000, 227],
  [14000, 159],
  [12000, 136],
  [6969, 79],
  [4200, 48],
  [69420, 789]
])

samples = metropolis_sample_from_posterior(p_prior, p_likelihood, measurements, x_0 = [25])

for i in range(len(samples)):
    chain = [v[0] for v in samples[i]]
    samples[i] = chain

all_samples = []

for chain in samples:
    all_samples += chain
    
mean = np.mean(all_samples)
stddev = np.std(all_samples)

cols = ["blue"]
i = 0

for chain in samples:
    col = cols[i]
    i = i + 1 if i + 1 < len(cols) else 0
    plt.hist(chain, density = True, bins=20, alpha = 1/len(samples), color=col)

print(mean, stddev)
x = np.linspace(min(all_samples), max(all_samples), 100)
y = normal_dist(x, mean, stddev)

plt.plot(x, y, lw = 4, color = "red")

plt.show()