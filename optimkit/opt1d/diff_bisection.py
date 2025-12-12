import numpy as np
import sympy as sp
from typing import Tuple

def diff_bisection(
        f_sym: sp.Expr, 
        alpha: float, 
        beta: float, 
        l: float = 1e-5, 
        tol: float = 1e-8
        )-> tuple[np.ndarray, np.ndarray,int]:
    """
    Derivative-based bisection method for 1D optimization.


    Parameters
    ----------
    f_sym : sympy expression
        Symbolic function to optimize.
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


    # symbolic variable
    var = list(f_sym.free_symbols)[0]
    # derivative
    df_sym = sp.diff(f_sym, var)
    df = sp.lambdify(var, df_sym, 'numpy')
    # number of iterations
    n = int(np.ceil(-np.log(l / (beta - alpha)) / np.log(2)))
    a = np.zeros(n)
    b = np.zeros(n)

    a[0] = alpha
    b[0] = beta
    for k in range(n - 1):
        mid = 0.5 * (a[k] + b[k])
        val = df(mid)
        if abs(val) < tol:
            # stop early
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

