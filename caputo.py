from math import gamma
import numpy as np

def caputo_euler(f, alpha, t_span, num_step, y0):
    """Integrate a system of Caputo fractional-order ODEs using the forward Euler method.
    
    Args:
        f (function): A scalar- or vector-valued fractional-order ODE, using the convention D_alpha[y] = f(t, y).
        alpha (float): The order of the derivative, in the range (0.0, 1.0]
        t_span (tuple): A pair of floats representing the inital and final times.
        num_step (float): The number of time steps used for integration.
        y0 (float or numpy.ndarray): A scalar or vector initial condition consistent with the FDE output.
    
    Returns:
        A tuple of numpy.ndarrays (ts, ys) consisting of the times and corresponding solution vectors. ts has dimension k + 1. ys has dimension k + 1 for scalar FDEs or (k + 1, n) for n-dimensional vector FDEs.
    """
    h = (t_span[1] - t_span[0]) / num_step
    ts = np.linspace(t_span[0], t_span[1], num_step + 1)
    if isinstance(y0, np.ndarray):
        # Initial conditions
        ys = np.tile(y0, (num_step + 1, 1))
        
        # Solution loop
        for j in range(1, num_step + 1):
            for k in range(j, num_step + 1):
                ys[k, :] += h ** alpha / gamma(alpha + 1) * ((k + 1 - j) ** alpha - (k - j) ** alpha) * f(ts[j], ys[j, :])
    else:
        ys = y0 * np.ones(num_step + 1)
        
        # Solution loop
        for j in range(0, num_step + 1):
            for k in range(j + 1, num_step + 1):
                ys[k] += h ** alpha / gamma(alpha + 1) * ((k + 1 - j) ** alpha - (k - j) ** alpha) * f(ts[j], ys[j])
    
    return ts, ys