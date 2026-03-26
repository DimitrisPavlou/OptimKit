import numpy as np
from typing import Callable, Tuple

NumericFunction = Callable[[np.ndarray], float]


def golden_section_search(
    g: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-6
) -> float:
    """
    Golden-section search for univariate function minimization.

    Args:
        g: Univariate function to minimize
        a: Lower bound of search interval
        b: Upper bound of search interval
        tol: Tolerance for convergence (default: 1e-6)

    Returns:
        Approximate minimizer in [a, b]
    """
    phi = (np.sqrt(5) - 1) / 2  # Golden ratio - 1

    while b - a > tol:
        x1 = b - phi * (b - a)
        x2 = a + phi * (b - a)

        if g(x1) < g(x2):
            b = x2
        else:
            a = x1

    return 0.5 * (a + b)


def _make_array_callable(f_numeric: Callable) -> Callable[[np.ndarray], float]:
    """
    Create a wrapper that ensures f_numeric is called with array argument.
    
    Handles both:
    - Symbolic functions: f_numeric(*x) - expects unpacked arguments
    - Numeric functions: f_numeric(x) - expects array directly
    
    Args:
        f_numeric: The numeric callable (may expect unpacked or packed arguments)
    
    Returns:
        A callable that accepts array argument: f(x)
    """
    def wrapped(x: np.ndarray) -> float:
        try:
            # Try calling with array directly (numeric functions)
            return float(f_numeric(x))
        except (TypeError, ValueError):
            # Fall back to unpacking (symbolic multivariate functions)
            return float(f_numeric(*x))
    return wrapped


def armijo_line_search(
    f_numeric: NumericFunction,
    xk: np.ndarray,
    dk: np.ndarray,
    grad: np.ndarray,
    gamma_init: float = 1.0,
    c1: float = 1e-4,
    rho: float = 0.5
) -> float:
    """
    Armijo backtracking line search for step size determination.

    Args:
        f_numeric: Objective function (takes array argument: f_numeric(x))
        xk: Current iterate
        dk: Search direction
        grad: Gradient at current iterate
        gamma_init: Initial step size (default: 1.0)
        c1: Armijo constant for sufficient decrease (default: 1e-4)
        rho: Backtracking factor (default: 0.5)

    Returns:
        Step size satisfying Armijo condition
    """
    # Wrap to handle both symbolic (*x) and numeric (x) calling conventions
    f_callable = _make_array_callable(f_numeric)
    
    gamma = gamma_init
    f_xk = f_callable(xk)
    directional_derivative = np.dot(grad, dk)

    # Backtrack until Armijo condition is satisfied
    while f_callable(xk + gamma * dk) > f_xk + c1 * gamma * directional_derivative:
        gamma *= rho

    return gamma


def optimal_line_search(
    f_numeric: NumericFunction,
    xk: np.ndarray,
    dk: np.ndarray,
    interval: Tuple[float, float] = (0.0, 10.0),
    tol: float = 1e-6
) -> float:
    """
    Optimal line search using golden-section search.

    Args:
        f_numeric: Objective function (takes array argument: f_numeric(x))
        xk: Current iterate
        dk: Search direction
        interval: Search interval [a, b] (default: (0.0, 10.0))
        tol: Tolerance for golden-section search (default: 1e-6)

    Returns:
        Optimal step size in the given interval
    """
    # Wrap to handle both symbolic (*x) and numeric (x) calling conventions
    f_callable = _make_array_callable(f_numeric)
    g = lambda gamma: f_callable(xk + gamma * dk)
    return golden_section_search(g, interval[0], interval[1], tol)