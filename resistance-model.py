import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

from matplotlib.widgets import Slider

trials = np.array(range(7))

r_exp = 5

v_exp = np.array([
    1,
    2,
    3,
    4,
    5,
    6,
    7
])

# [v, i]
observed = np.array([
    [1, 0.194],
    [2, 0.39],
    [3, 0.585],
    [4, 0.78],
    [5, 0.975],
    [6, 1.17],
    [7, 1.366]
])

s_v = float( 0.1 )
s_i = float( 0.1 )
s_r = float( 2 )

def normal_dist(x, u, s):
    density = 1 / ( s * np.sqrt( 2 * np.pi ) ) * np.exp( -1/2 * ( ( x - u ) / s ) ** 2 )

    return density

def p_prior(x):
    density = normal_dist(x, r_exp, s_r)

    # if x >= 0 and x <= 10:
    #     density = 1/10
    # else:
    #     density = 0

    return density

def p_likelihood(x):
    trial_densities = []

    i_exp = v_exp / x

    for n in trials:
        v = observed[n][0]
        i = observed[n][1]

        trial_densities.append( normal_dist( v, v_exp[n], s_v ) * normal_dist( i, i_exp[n], s_i ) )
    
    return np.prod( trial_densities )

p_evidence = integrate.quad( lambda x: p_likelihood(x) * p_prior(x), max( r_exp - 10*s_r, 0.01 ), r_exp + 10*s_r ) [0]

x = np.arange(0, 10, 0.05)
def f(x):
    y = x.copy()

    p_evidence = integrate.quad( lambda x: p_likelihood(x) * p_prior(x), max( r_exp - 10*s_r, 0.01 ), r_exp + 10*s_r ) [0]

    for i in range( len( y ) ):
        y[i] = p_likelihood(x[i]) * p_prior(x[i]) / p_evidence
    
    return y
y = f(x)

r = x[np.argmax(y)]
print(r)

fig, ax = plt.subplots()
line, = ax.plot( x, y, lw=2 )
ax.set_ylim(0, max(y) + 0.1)

fig.subplots_adjust(bottom = 0.25)

axr = fig.add_axes([0.24, 0.1, 0.65, 0.03])
r_slider = Slider(
    ax=axr,
    label='p_prior mean',
    valmin=1,
    valmax=10,
    valinit=r_exp,
)
axs_r = fig.add_axes([0.24, 0.04, 0.65, 0.03])
s_r_slider = Slider(
    ax=axs_r,
    label='p_prior std',
    valmin=0.01,
    valmax=1.5,
    valinit=s_r,
)

def update(val):
    global r_exp, s_r, i_exp
    r_exp = r_slider.val
    s_r = s_r_slider.val

    i_exp = v_exp / r_exp

    y = f(x)

    line.set_ydata(y)

    ax.set_ylim(0, max(0.5, max(y) + 0.1))

r_slider.on_changed(update)
s_r_slider.on_changed(update)

z = x.copy()

for i in range( len( z ) ):
    z[i] = p_likelihood(x[i])

ax.plot(x, z / max(z))

z2 = x.copy()

for i in range( len( z ) ):
    z2[i] = p_prior(x[i])

ax.plot(x, z2)

plt.show()