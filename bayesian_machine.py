import numpy as np
from numpy import random

def normal_dist(x, u, s):
    density = 1 / ( s * np.sqrt( 2 * np.pi ) ) * np.exp( -1/2 * ( ( x - u ) / s ) ** 2 )

    return density

def uniform(x, a, b):
    if x >= a and x <= b:
        return 1/(b-a)
    else:
        return 0

def multivar_dist(x, u, s):
    k = len(x)
    
    density = (2*np.pi)**(-k/2) * np.linalg.det(s)**(-1/2)
    density *= np.exp(-1/2 * (x-u).T * np.linalg.inv(s) * (x-u))
    
    return density

def metropolis_sample_from_posterior(p_prior, 
                                     p_likelihood, 
                                     measurements, 
                                     num_chains=4, 
                                     num_samples=2500, 
                                     warmup_samples=1250, 
                                     x_0=[0],
                                     sample_jump = [[1]]):
    samples = []
    
    k = len(x_0)
    
    for j in range(num_chains):
        chain = []
        
        x = x_0
        p = p_prior(np.array([x])) * p_likelihood(np.array([x]), measurements)
        for i in range(num_samples):
            new_x = x + random.multivariate_normal(np.array([0]*k), sample_jump)
            
            new_p = p_prior(np.array([new_x])) * p_likelihood(np.array([new_x]), measurements)
            
            move_chance = min(1, new_p / p)
            
            if random.rand() < move_chance:
                x = new_x
                p = new_p
                
            if i > warmup_samples:
                chain.append(x)

        samples.append(chain)
        
    return samples