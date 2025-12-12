import numpy as np
from sympy import Expr, lambdify
from typing import Tuple


def golden_sector(
    f: Expr,
    alpha: float,
    beta: float,
    length_tol: float
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Perform the Golden Section search on a symbolic function.

    Parameters
    ----------
    f : sympy.Expr
        Symbolic expression of the objective function.
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

    vars = list(f.free_symbols)
    f_numeric = lambdify(vars, f, "numpy")

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
        val1 = f_numeric(x1[k])
        val2 = f_numeric(x2[k])
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
