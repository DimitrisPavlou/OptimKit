import numpy as np
from sympy import Expr, lambdify
from typing import Tuple


def fibonacci_numbers(n: int) -> np.ndarray:
    """
    Return Fibonacci numbers from F_0 to F_n.
    """
    fib = np.zeros(n + 1, dtype=float)
    fib[0] = 0
    fib[1] = 1
    for i in range(2, n + 1):
        fib[i] = fib[i - 1] + fib[i - 2]
    return fib


def fibonacci(
    f: Expr,
    alpha: float,
    beta: float,
    length_tol: float,
    epsilon: float
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Fibonacci search method for 1D optimization.

    Parameters
    ----------
    f : sympy.Expr
        Symbolic objective function.
    alpha : float
        Initial interval lower bound.
    beta : float
        Initial interval upper bound.
    length_tol : float
        Stopping tolerance.
    epsilon : float
        Small positive number used for the final step.

    Returns
    -------
    low : np.ndarray
        Array of interval lower bounds.
    high : np.ndarray
        Array of interval upper bounds.
    num_operations : int
        Number of iterations (≈ number of function evaluations).
    """

    vars = list(f.free_symbols)
    f_numeric = lambdify(vars, f, "numpy")

    val = (beta - alpha) / length_tol
    n = 1 + int(np.ceil(np.log(np.sqrt(5) * val) / np.log(1.618)))

    a = np.zeros(n)
    b = np.zeros(n)
    x1 = np.zeros(n)
    x2 = np.zeros(n)

    F = fibonacci_numbers(n)

    a[0] = alpha
    b[0] = beta
    x1[0] = a[0] + (F[n - 2] / F[n]) * (b[0] - a[0])
    x2[0] = a[0] + (F[n - 1] / F[n]) * (b[0] - a[0])

    val1 = f_numeric(x1[0])
    val2 = f_numeric(x2[0])

    for k in range(n - 2):

        if val1 > val2:
            a[k + 1] = x1[k]
            b[k + 1] = b[k]

            x1[k + 1] = x2[k]
            x2[k + 1] = a[k + 1] + (F[n - k - 1] / F[n - k]) * (b[k + 1] - a[k + 1])

            val1 = val2
            val2 = f_numeric(x2[k + 1])

        else:
            a[k + 1] = a[k]
            b[k + 1] = x2[k]

            x2[k + 1] = x1[k]
            x1[k + 1] = a[k + 1] + (F[n - k - 2] / F[n - k]) * (b[k + 1] - a[k + 1])

            val2 = val1
            val1 = f_numeric(x1[k + 1])

    # Final epsilon-based decision
    x1[n - 1] = x1[n - 2]
    x2[n - 1] = x1[n - 2] + epsilon

    if f_numeric(x1[n - 1]) > f_numeric(x2[n - 1]):
        a[n - 1] = x1[n - 1]
        b[n - 1] = b[n - 2]
    else:
        a[n - 1] = a[n - 2]
        b[n - 1] = x2[n - 1]

    return a, b, n
