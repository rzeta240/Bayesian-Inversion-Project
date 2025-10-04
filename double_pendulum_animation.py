import numpy as np

import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

g = 9.81

m_1 = 1
m_2 = 1
l_1 = 1
l_2 = 1

mu = m_2 / m_1
lam = l_2 / l_1
tau = np.sqrt(g/l_1)

theta1_0 = np.radians(90)
theta2_0 = np.radians(20)
omega1_0 = np.radians(0)
omega2_0 = np.radians(0)

def double_pendulum(t, y):
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

t_max = tau*20

t_eval = np.arange(0, t_max, 0.02)

sol = solve_ivp(double_pendulum, (0, t_max), [theta1_0, theta2_0, omega1_0, omega2_0], t_eval=t_eval)

print(sol)

# chatgpt code below

theta1, theta2 = sol.y[0], sol.y[1]

x1 = l_1 * np.sin(theta1)
y1 = -l_1 * np.cos(theta1)

x2 = x1 + l_2 * np.sin(theta2)
y2 = y1 - l_2 * np.cos(theta2)

from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()
ax.set_xlim(- (l_1+l_2) * 1.1, (l_1+l_2) * 1.1)
ax.set_ylim(- (l_1+l_2) * 1.1, (l_1+l_2) * 1.1)
ax.set_aspect('equal')

line, = ax.plot([], [], 'o-', lw=2)

def init():
    line.set_data([], [])
    return line,

def update(i):
    thisx = [0, x1[i], x2[i]]
    thisy = [0, y1[i], y2[i]]
    line.set_data(thisx, thisy)
    return line,

td = (sol["t"][1] - sol["t"][0]) / tau * 1000

ani = FuncAnimation(fig, update, frames=len(t_eval), init_func=init, blit=True, interval=td)

# from matplotlib.animation import PillowWriter

# writer = PillowWriter(fps=30)
# ani.save("double_pendulum.gif", writer=writer)

plt.show()