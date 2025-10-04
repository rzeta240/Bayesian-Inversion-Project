import numpy as np
import matplotlib.pyplot as plt

r_exp = 5

measurements = np.array([
    [1, 0.194],
    [2, 0.39],
    [3, 0.585],
    [4, 0.78],
    [5, 0.975],
    [6, 1.17],
    [7, 1.366]
])

v_exp = np.array([
    1,
    2,
    3,
    4,
    5,
    6,
    7
])

s_v = float( 0.1 )
s_i = float( 0.1 )
s_r = float( 2 )

def normal_dist(x, u, s):
    density = 1 / ( s * np.sqrt( 2 * np.pi ) ) * np.exp( -1/2 * ( ( x - u ) / s ) ** 2 )

    return density

def p_prior(u):
    density = normal_dist(u, r_exp, s_r)
    
    return density

def p_likelihood(u, d):
    density = []
    
    i_exp = v_exp / u[0]
    
    for i in range( len( d ) ):
        density.append( normal_dist( d[i][0], v_exp[i], s_v ) * normal_dist( d[i][1], i_exp[i], s_i ) )
        
    return np.prod(density)

from bayesian_machine import metropolis_sample_from_posterior

samples = metropolis_sample_from_posterior(p_prior, p_likelihood, measurements, x_0 = [r_exp])

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