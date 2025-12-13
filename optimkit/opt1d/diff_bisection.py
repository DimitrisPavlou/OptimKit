import numpy as np
import sympy as sp
from typing import Tuple
from optimkit.function.Function import Function

def diff_bisection(
    f: Function, 
    alpha: float, 
    beta: float, 
    l: float = 1e-5, 
    tol: float = 1e-8
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Derivative-based bisection method for 1D optimization using a Function object.
    
    Parameters
    ----------
    f : Function
        Function object (must be univariate, n_vars=1, symbolic type).
    alpha, beta : float
        Interval bounds.
    l : float
        Desired precision.
    tol : float
        Tolerance for derivative near zero.
    
    Returns
    -------
    low : np.ndarray
        The sequence of lower bounds (a_k).
    high : np.ndarray
        The sequence of upper bounds (b_k).
    num_operations : int
        Number of derivative evaluations (same as number of iterations).
    """
    if f.n_vars != 1:
        raise ValueError("Derivative bisection requires a univariate function (n_vars=1)")
    if f.func_type != "symbolic":
        raise ValueError("Derivative bisection requires a symbolic function")
    
    # Calculate number of iterations
    n = int(np.ceil(-np.log(l / (beta - alpha)) / np.log(2)))
    
    a = np.zeros(n)
    b = np.zeros(n)
    a[0] = alpha
    b[0] = beta
    
    for k in range(n - 1):
        mid = 0.5 * (a[k] + b[k])
        val = f.grad(mid)
        
        if abs(val) < tol:
            # Stop early
            a[k+1:] = a[k]
            b[k+1:] = b[k]
            break
        elif val > 0:
            a[k+1] = a[k]
            b[k+1] = mid
        else:
            a[k+1] = mid
            b[k+1] = b[k]
    
    return a, b, n
