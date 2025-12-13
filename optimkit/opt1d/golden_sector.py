import numpy as np
from sympy import Expr, lambdify
from typing import Tuple
from optimkit.function.Function import Function

def golden_sector(
    f: Function,
    alpha: float,
    beta: float,
    length_tol: float
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Perform the Golden Section search using a Function object.
    
    Parameters
    ----------
    f : Function
        Function object (must be univariate, n_vars=1).
    alpha : float
        Initial lower bound.
    beta : float
        Initial upper bound.
    length_tol : float
        Stopping threshold on interval size.
    
    Returns
    -------
    low : np.ndarray
        Lower interval bound sequence.
    high : np.ndarray
        Upper interval bound sequence.
    num_operations : int
        Number of function evaluations.
    """
    if f.n_vars != 1:
        raise ValueError("Golden section search requires a univariate function (n_vars=1)")
    
    max_iter = 5000
    a = np.zeros(max_iter)
    b = np.zeros(max_iter)
    x1 = np.zeros(max_iter)
    x2 = np.zeros(max_iter)
    
    gamma = 0.618
    a[0] = alpha
    b[0] = beta
    x1[0] = a[0] + (1 - gamma) * (b[0] - a[0])
    x2[0] = a[0] + gamma * (b[0] - a[0])
    
    k = 0
    num_ops = 0
    
    while (b[k] - a[k] > length_tol) and (k < max_iter - 1):
        val1 = f(x1[k])
        val2 = f(x2[k])
        num_ops += 2
        
        if val1 > val2:
            a[k + 1] = x1[k]
            b[k + 1] = b[k]
            x1[k + 1] = x2[k]
            x2[k + 1] = a[k + 1] + gamma * (b[k + 1] - a[k + 1])
        else:
            a[k + 1] = a[k]
            b[k + 1] = x2[k]
            x2[k + 1] = x1[k]
            x1[k + 1] = a[k + 1] + (1 - gamma) * (b[k + 1] - a[k + 1])
        k += 1
    
    return a[:k + 1], b[:k + 1], num_ops