import numpy as np

"""
u_xx + k^2 * u = 0
u_x(0) = ga, u_x(1) = gb
"""

def Analytical(k):
    u = lambda x: - np.cos(k * x) / (k * np.sin(k))
    u_x = lambda x: np.sin(k * x) / np.sin(k)
    return (u, u_x)
