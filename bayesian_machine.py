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

def log_normal(x, u, s):
    density = np.log(1 / ( s * np.sqrt( 2 * np.pi ) )) + -1/2 * ( ( x - u ) / s ) ** 2
    
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
    
    print(f"{f"Sampling progress:":<50}|")
    
    for j in range(num_chains):
        print(f"{f"Chain {j+1}":<50}|")
        
        chain = []
        
        if x_0[0] == "random":
            x = np.array([random.rand()*(x_0[2] - x_0[1])+x_0[1]])
        else:
            x = np.array(x_0)
        
        k = len(x)
        
        p = p_prior(np.array([x])) * p_likelihood(x, measurements)
        
        progress = 0
        for i in range(num_samples):
            if i >= progress + 0.02*num_samples:
                progress += 0.02*num_samples
                print("*", end="", flush=True)
            if i+1 == num_samples:
                print("*|")
            
            new_x = x + random.multivariate_normal(np.array([0]*k), sample_jump)
            
            new_p = p_prior(np.array([new_x]))
            
            if not new_p == 0:
                new_p *= p_likelihood(new_x, measurements)
            
            move_chance = min(1, new_p / p)
            
            if random.rand() < move_chance:
                x = new_x
                p = new_p
                
            if i > warmup_samples:
                chain.append(x)

        samples.append(chain)
        
    return samples

def log_metropolis(p_prior, 
                   p_likelihood, 
                   measurements, 
                   num_chains=4, 
                   num_samples=2500, 
                   warmup_samples=1250, 
                   x_0=[0],
                   sample_jump = [[1]]):
    samples = []
    
    print(f"{f"Sampling progress:":<50}|")
    
    for j in range(num_chains):
        print(f"{f"Chain {j+1}":<50}|")
        
        chain = []
        
        if x_0[0] == "random":
            x = np.array([random.rand()*(x_0[2] - x_0[1])+x_0[1]])
        else:
            x = np.array(x_0)
        
        k = len(x)
        
        p = p_prior(np.array([x])) + p_likelihood(x, measurements)
        
        progress = 0
        for i in range(num_samples):
            if i >= progress + 0.02*num_samples:
                progress += 0.02*num_samples
                print("*", end="", flush=True)
            if i+1 == num_samples:
                print("*|")
            
            new_x = x + random.multivariate_normal(np.array([0]*k), sample_jump)
            
            new_p = p_prior(np.array([new_x]))
            
            if not new_p == 0:
                new_p += p_likelihood(new_x, measurements)
            
            move_chance = min(1, np.exp(new_p - p))
            
            if random.rand() < move_chance:
                x = new_x
                p = new_p
                
            if i > warmup_samples:
                chain.append(x)

        samples.append(chain)
        
    return samples