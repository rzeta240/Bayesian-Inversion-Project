import numpy as np
import matplotlib.pyplot as plt

from bayesian_machine import multivar_dist, normal_dist, metropolis_sample_from_posterior, uniform

r_exp = 9.99

measurements = np.array([
    [0.997, 0.098],
    [1.978, 0.197],
    [2.96, 0.296],
    [3.95, 0.395],
    [4.94, 0.494],
    [5.93, 0.593],
    [6.92, 0.692],
    [7.91, 0.791],
    [8.89, 0.889],
    [9.88, 0.988]
])

# measurements = np.array([
#     [1, 0.194],
#     [2, 0.39],
#     [3, 0.585],
#     [4, 0.78],
#     [5, 0.975],
#     [6, 1.17],
#     [7, 1.366]  
# ])

s_d = float( 0.1 )
s_r = float( 2 )

def p_prior(u):
    density = uniform(u, 1, 20)
    
    return density

def p_likelihood(u, d):
    density = []
    
    for di in d:
        distance = np.abs(u * di[1] - di[0]) / np.sqrt(1 + u**2)
        
        density.append( normal_dist( distance, 0, s_d ) )
        
    return np.prod(density)

samples = metropolis_sample_from_posterior(p_prior, p_likelihood, measurements, x_0 = [r_exp], sample_jump=[[s_d]])

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

print(mean, stddev) # Get uncertainty for the fitting 
x = np.linspace(min(all_samples), max(all_samples), 100)
y = normal_dist(x, mean, stddev)

plt.plot(x, y, lw = 4, color = "red")

plt.show()