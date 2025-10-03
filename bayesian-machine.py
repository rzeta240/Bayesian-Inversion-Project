import numpy as np

# Given the measurements

d = np.array([
    [], # Measurement 1
    [] # Measurement 2
])

# Given p_prior

def p_prior(u: np.array):
    p = 1
    
    return p # Density

# Given p_likelihood

def p_likelihood(d: np.array, u: np.array):
    p = 1
    
    return p # Density

# Get p_posterior, proportional to p_prior and p_likelihood

print(p_prior(np.array([])))
print(p_likelihood(np.array([]), np.array([])))