import numpy as np
from sympy import Expr, lambdify, symbols
from typing import Tuple
from optimkit.function.Function import Function


def bisection(
    f: Function, 
    alpha: float, 
    beta: float, 
    length_tol: float, 
    epsilon: float
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Perform 1D bisection search using a Function object.
    
    Parameters
    ----------
    f : Function
        Function object (must be univariate, n_vars=1).
    alpha : float
        Initial lower bound of the interval.
    beta : float
        Initial upper bound of the interval.
    length_tol : float
        Stop when interval length <= length_tol.
    epsilon : float
        Small positive shift used to compare f(mid - ε) and f(mid + ε).
    
    Returns
    -------
    low : np.ndarray
        Array of all lower interval bounds a(k).
    high : np.ndarray
        Array of all upper interval bounds b(k).
    num_operations : int
        Number of function evaluations performed.
    """
    if f.n_vars != 1:
        raise ValueError("Bisection method requires a univariate function (n_vars=1)")
    
    max_iter = 5000
    a = np.zeros(max_iter)
    b = np.zeros(max_iter)
    a[0] = alpha
    b[0] = beta
    k = 0
    num_ops = 0
    
    while (b[k] - a[k] > length_tol) and (k < max_iter - 1):
        mid = 0.5 * (a[k] + b[k])
        x1 = mid - epsilon
        x2 = mid + epsilon
        val1 = f(x1)
        val2 = f(x2)
        num_ops += 2
        
        if val1 < val2:
            a[k + 1] = a[k]
            b[k + 1] = x2
        else:
            a[k + 1] = x1
            b[k + 1] = b[k]
        k += 1
    
    return a[:k + 1], b[:k + 1], num_ops
