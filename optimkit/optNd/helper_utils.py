import numpy as np
from sympy import Expr, lambdify, symbols, Matrix
from typing import Callable, Union, Tuple, List

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
        f_numeric: Objective function
        xk: Current iterate
        dk: Search direction
        grad: Gradient at current iterate
        gamma_init: Initial step size (default: 1.0)
        c1: Armijo constant for sufficient decrease (default: 1e-4)
        rho: Backtracking factor (default: 0.5)
        
    Returns:
        Step size satisfying Armijo condition
    """
    gamma = gamma_init
    f_xk = f_numeric(*xk)
    directional_derivative = np.dot(grad, dk)
    
    # Backtrack until Armijo condition is satisfied
    while f_numeric(*(xk + gamma * dk)) > f_xk + c1 * gamma * directional_derivative:
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
        f_numeric: Objective function
        xk: Current iterate
        dk: Search direction
        interval: Search interval [a, b] (default: (0.0, 10.0))
        tol: Tolerance for golden-section search (default: 1e-6)
        
    Returns:
        Optimal step size in the given interval
    """
    g = lambda gamma: f_numeric(*(xk + gamma * dk))
    return golden_section_search(g, interval[0], interval[1], tol)
