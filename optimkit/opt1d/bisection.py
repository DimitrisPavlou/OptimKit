import numpy as np
from sympy import Expr, lambdify, symbols
from typing import Tuple


def bisection(
    f: Expr, 
    alpha: float, 
    beta: float, 
    length_tol: float, 
    epsilon: float
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Perform 1D bisection search on a symbolic function using a small epsilon
    to approximate derivative-like behavior.

    Parameters
    ----------
    f : sympy.Expr
        Symbolic expression of the objective function.
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

    # Convert symbolic function to numeric function
    vars = list(f.free_symbols)
    f_numeric = lambdify(vars, f, "numpy")

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

        val1 = f_numeric(x1)
        val2 = f_numeric(x2)
        num_ops += 2

        if val1 < val2:
            a[k + 1] = a[k]
            b[k + 1] = x2
        else:
            a[k + 1] = x1
            b[k + 1] = b[k]

        k += 1

    return a[:k + 1], b[:k + 1], num_ops
