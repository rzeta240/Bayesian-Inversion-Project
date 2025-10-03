import numpy as np

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