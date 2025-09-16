import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

m_1 = 1
m_2 = 1
l_1 = 1
l_2 = 1

theta1_0 = np.radians(20)
theta2_0 = np.radians(20)

def double_pendulum(t, y):
    theta1, theta2, omega1, omega2 = y

    

# def double_pendulum(t, y):
#     theta1, theta2, omega1, omega2 = y

#     delta = theta2 - theta1
#     denom1 = (m_1 + m_2) * l_1 - m_2 * l_1 * np.cos(delta) ** 2
#     denom2 = (l_2 / l_1) * denom1

#     dtheta1 = omega1
#     dtheta2 = omega2

#     domega1 = (
#                       m_2 * l_1 * omega1**2 * np.sin(delta) * np.cos(delta)
#                       + m_2 * g * np.sin(theta2) * np.cos(delta)
#                       + m_2 * l_2 * omega2**2 * np.sin(delta)
#                       - (m_1 + m_2) * g * np.sin(theta1)
#               ) / denom1

#     domega2 = (
#                       -m_2 * l_2 * omega2**2 * np.sin(delta) * np.cos(delta)
#                       + (m_1 + m_2) * g * np.sin(theta1) * np.cos(delta)
#                       - (m_1 + m_2) * l_1 * omega1**2 * np.sin(delta)
#                       - (m_1 + m_2) * g * np.sin(theta2)
#               ) / denom2

#     return [dtheta1, dtheta2, domega1, domega2]