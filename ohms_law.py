import numpy as np
import matplotlib.pyplot as plt

from bayesian_machine import normal_dist, metropolis_sample_from_posterior, uniform, log_metropolis, log_normal

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

# v_exp = np.array([
#   1,
#   2,
#   3,
#   4,
#   5,
#   6,
#   7,
#   8,
#   9,
#   10  
# ])

v_exp = np.array([di[0] for di in measurements])

def get_i_exp(r):
    return v_exp / r

# measurements = np.array([
#     [1, 0.194],
#     [2, 0.39],
#     [3, 0.585],
#     [4, 0.78],
#     [5, 0.975],
#     [6, 1.17],
#     [7, 1.366]  
# ])

s_v = float( 1 )
s_i = float( 0.01 )
s_r = float( 2 )

def p_prior(u):
    density = log_normal( u, r_exp, s_r )
    
    return density

def p_likelihood(u, d):
    density = []
    
    i_exp = get_i_exp(u)
    
    for i in range( len( d ) ):
        di = d[i]
        vi = v_exp[i]
        ii = i_exp[i]
        
        density.append( log_normal( di[0], vi, s_v ) + log_normal( di[1], ii, s_i ) )
        
    return np.sum(density)

samples = log_metropolis(p_prior, p_likelihood, measurements, x_0 = ["random", 5, 15], sample_jump=[[np.sqrt(s_v*s_i)]], warmup_samples=5000, num_samples=10000)

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