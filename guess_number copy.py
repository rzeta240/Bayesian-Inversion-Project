import matplotlib.pyplot as plt
import numpy as np

from bayesian_machine import metropolis_sample_from_posterior, normal_dist

def p_prior(u):
    u = u[0]
    
    if u >= 0 and u < 10:
        return 1/10
    else:
        return 0

def p_likelihood(u, d):
    densities = []
    
    for i in range(len(d)):
        densities.append(normal_dist(d[i][1], d[i][0]**(1/u[0]), 300))
    
    return np.prod(densities)

measurements = np.array([
  [7000000, 10],
  [9000000, 10],
  [5000000, 9],
  [17000000, 11],
  [17*10**9, 29],
  [23*10**9, 30],
  [139*10**9, 39],
  [230*10**20, 1565]
])

samples = metropolis_sample_from_posterior(p_prior, p_likelihood, measurements, x_0 = [5])

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