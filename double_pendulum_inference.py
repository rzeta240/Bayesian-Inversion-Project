import numpy as np
from numpy import random

import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from bayesian_machine import normal_dist, uniform, metropolis_sample_from_posterior

import time
start_time = time.time()

g = 9.81

m_1 = 1
m_2 = 1.7
l_1 = 1
l_2 = 1

mu = m_2 / m_1
lam = l_2 / l_1
tau = np.sqrt(g/l_1)

theta1_0 = np.radians(90)
theta2_0 = np.radians(20)
omega1_0 = np.radians(0)
omega2_0 = np.radians(0)

def double_pendulum(t, y, mu, lam):
    theta1, theta2, omega1, omega2 = y

    dtheta1 = omega1
    dtheta2 = omega2

    M = np.array([[1+mu, mu*lam*np.cos(theta1-theta2)],
        [mu*lam*np.cos(theta1-theta2), mu*lam**2]])
    f = np.array([[-(1+mu)*np.sin(theta1) - mu*lam*(np.sin(theta1-theta2))*dtheta2**2],
        [mu*lam*(-np.sin(theta2) + (np.sin(theta1-theta2))*dtheta1**2)]])

    u = np.matmul(np.linalg.inv(M), f)

    domega1, domega2 = u[0][0], u[1][0]

    return [dtheta1, dtheta2, domega1, domega2]


def take_measurements(mu, lam):

    t_max = tau*5

    t_eval = np.arange(0, t_max, 0.02)

    sol = solve_ivp(lambda t, y: double_pendulum(t, y, mu, lam), (0, t_max), [theta1_0, theta2_0, omega1_0, omega2_0], t_eval=t_eval, method='BDF')

    measurements_theta = [
        [sol["y"][j][i*50] for j in range(2)] for i in range(1, 6)
    ]

    measurements = []

    for i in range(len(measurements_theta)):
        m = measurements_theta[i]

        x1 = np.cos(m[0]) * l_1
        y1 = np.sin(m[0]) * l_1

        x2 = x1 + np.cos(m[1]) * l_2
        y2 = y1 + np.sin(m[1]) * l_2

        measurements.append([i+1, x1, y1, x2, y2])
    
    return measurements

# Generate "measurements"

measurements = take_measurements(mu, lam)

# for m in measurements:
#     print(m)

for j in range(len( measurements )):
    m = measurements[j]

    new_m = [m[0], 0, 0, 0, 0]
    for i in range(1, 5):
        new_m[i] = m[i] + random.normal(scale = 0.1)
    measurements[j] = new_m

# for m in measurements:
#     print(m)

# Define densities. Let's guess mu, assuming lam is known

def p_prior(u):
    density = uniform(u, 1/10, 10)

    return density

meas_error = 0.2

def p_likelihood(u, d):
    u = u[0]

    expected = take_measurements(u, lam)

    densities = []

    for i in range(len(d)):
        di = d[i]
        ei = expected[i]

        for j in range(4):
            densities.append( normal_dist(di[j], ei[j], meas_error) )

    return np.prod(densities)

# Get p_posterior

samples = metropolis_sample_from_posterior(p_prior, p_likelihood, measurements, x_0 = [1],
                                           num_chains=4,
                                           num_samples=5000,
                                           warmup_samples=2500)

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

end_time = time.time()

print(f"Execution time: {end_time - start_time:.3f}s")

plt.show()