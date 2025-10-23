from math import gamma
import numpy as np

def caputo_l1_diff(fs, alpha, dt=1.0, ts=None):
    """Compute approximate Caputo fractional derivatice using the L1 scheme.

    Args:
        fs (np.ndarray): A series of function values. For vector-valued functions, dimension 0 is assumed to correspond to time.
        alpha: The order of the derivative, in the range (0.0, 1.0].
        dt=1.0 (float): If the corresponding dependent variable series is not provided, the uniform step size between function evaluations.
        ts=None (np.ndarray): The corresponding dependent variable series, used if there is a variable step size.

    Returns:
        An np.ndarray containing the 
    """
    fps = np.zeros(fs.size)
    C = dt ** -alpha / gamma(2 - alpha)
    dfs = fs[1:] - fs[:-1]
    if ts == None:
        rs = np.arange(fps.size - 1)
        ws = (rs + 1) ** alpha - rs ** alpha # Convolution weights
        for k in range(fps.size - 1):
            fps[k + 1] = C * np.sum(dfs[:(k + 1)] * np.flip(ws[:(k + 1)]))
    else:
        raise('Variable time step differentiation not yet implemented!')
    return fps

def caputo_euler(f, alpha, t_span, num_step, y0):
    """Integrate a system of Caputo fractional-order ODEs using the forward Euler method.
    
    Args:
        f (function): A scalar- or vector-valued fractional-order ODE, using the convention D_alpha[y] = f(t, y).
        alpha (float): The order of the derivative, in the range (0.0, 1.0].
        t_span (tuple): A pair of floats representing the inital and final times.
        num_step (int): The number of time steps used for integration.
        y0 (float or numpy.ndarray): A scalar or vector initial condition consistent with the FDE output.
    
    Returns:
        A tuple of numpy.ndarrays (ts, ys) consisting of the times and corresponding solution vectors. ts has dimension k + 1. ys has dimension k + 1 for scalar FDEs or (k + 1, n) for n-dimensional vector FDEs.
    """
    h = (t_span[1] - t_span[0]) / num_step
    ts = np.linspace(t_span[0], t_span[1], num_step + 1)
    rs = np.arange(num_step)
    ws = (rs + 1) ** alpha - rs ** alpha # Convolution weights
    C = h ** alpha / gamma(alpha + 1)
    if isinstance(y0, np.ndarray):
        # Initial conditions
        ys = np.zeros((num_step + 1, y0.size))
        ys[0, :] = y0
        fs = np.zeros((num_step + 1, y0.size))

        # Solution loop
        for k in range(num_step):
            fs[k, :] = f(ts[k], ys[k, :])
            ys[k + 1, :] = y0 + C * np.sum(fs[:(k + 1), :] * np.flip(ws[:(k + 1)]).reshape(-1, 1), axis=0)
    else:
        # Initial conditions
        ys = np.zeros(num_step + 1)
        ys[0] = y0
        fs = np.zeros(num_step + 1)

        # Solution loop
        for k in range(num_step):
            fs[k] = f(ts[k], ys[k])
            ys[k + 1] = y0 + C * np.sum(fs[:(k + 1)] * np.flip(ws[:(k + 1)]))
    return ts, ys

def naive_caputo_euler(f, alpha, t_span, num_step, y0):
    """Integrate a system of Caputo fractional-order ODEs using the forward Euler method.
    
    Args:
        f (function): A scalar- or vector-valued fractional-order ODE, using the convention D_alpha[y] = f(t, y).
        alpha (float): The order of the derivative, in the range (0.0, 1.0].
        t_span (tuple): A pair of floats representing the inital and final times.
        num_step (int): The number of time steps used for integration.
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
        for j in range(0, num_step + 1):
            for k in range(j + 1, num_step + 1):
                ys[k, :] += h ** alpha / gamma(alpha + 1) * ((k - j) ** alpha - (k - j - 1) ** alpha) * f(ts[j], ys[j, :])
    else:
        # Initial conditions
        ys = y0 * np.ones(num_step + 1)
        
        # Solution loop
        for j in range(0, num_step + 1):
            for k in range(j + 1, num_step + 1):
                ys[k] += h ** alpha / gamma(alpha + 1) * ((k - j) ** alpha - (k - j - 1) ** alpha) * f(ts[j], ys[j])
    
    return ts, ys