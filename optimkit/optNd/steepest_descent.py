import numpy as np
from typing import Union, Tuple, List
from .helper_utils import armijo_line_search, optimal_line_search
from optimkit.function.Function import Function


def steepest_descent(
    f: Function,
    starting_point: Union[np.ndarray, List[float]],
    epsilon: float = 1e-6,
    gamma_selection: str = "armijo",
    gamma: float = 1.0,
    alpha: float = 1e-4,
    beta: float = 0.5,
    max_iter: int = 5000
) -> Tuple[np.ndarray, int, np.ndarray, np.ndarray]:
    """
    Steepest descent optimization algorithm for multivariable functions.
    
    Args:
        f: Function object (must be multivariate, n_vars > 1, symbolic type)
        starting_point: Initial point for optimization
        epsilon: Convergence tolerance for gradient norm (default: 1e-6)
        gamma_selection: Step size selection method - "armijo", "optimal_line_search", 
                        or "constant" (default: "armijo")
        gamma: Step size for "constant" method or initial step size for "armijo" (default: 1.0)
        alpha: Lower bound for "optimal_line_search" or c1 parameter for "armijo" (default: 1e-4)
        beta: Upper bound for "optimal_line_search" or rho parameter for "armijo" (default: 0.5)
        max_iter: Maximum number of iterations (default: 5000)
        
    Returns:
        Tuple containing:
        - xk: Array of iterates (shape: [n_iterations, n_variables])
        - n_iter: Number of iterations performed
        - grad_norms: Array of gradient norms at each iteration
        - f_vals: Array of function values at each iteration
        
    Raises:
        ValueError: If invalid gamma_selection method or missing required parameters
    """
    # Validate function type
    if f.n_vars < 2:
        raise ValueError("Steepest descent requires a multivariate function (n_vars >= 2)")
    if f.func_type != "symbolic":
        raise ValueError("Steepest descent requires a symbolic function")
    
    # Initialize
    xk = np.array(starting_point, dtype=float).flatten()
    n_vars = len(xk)
    
    if n_vars != f.n_vars:
        raise ValueError(f"Starting point dimension ({n_vars}) must match function variables ({f.n_vars})")
    
    # Storage for trajectory
    trajectory = [xk.copy()]
    grad = f.grad(xk)
    grad_norms = [np.linalg.norm(grad)]
    f_vals = [float(f(xk))]
    
    # Validate gamma_selection parameters
    interval: Tuple[float, float] = (0.0, 10.0)  # Default interval
    
    if gamma_selection == "optimal_line_search":
        interval = (alpha, beta)
    elif gamma_selection == "armijo":
        if gamma <= 0 or alpha <= 0 or alpha >= 1 or beta <= 0 or beta >= 1:
            raise ValueError(
                "For Armijo: gamma > 0, 0 < alpha < 1, 0 < beta < 1"
            )
    elif gamma_selection == "constant":
        if gamma <= 0:
            raise ValueError("For constant step size: gamma must be positive")
    else:
        raise ValueError(
            f"Unknown gamma_selection: '{gamma_selection}'. "
            "Choose from 'armijo', 'optimal_line_search', or 'constant'."
        )
    
    # Main optimization loop
    k = 0
    while grad_norms[-1] > epsilon and k < max_iter:
        # Search direction: negative gradient
        dk = -grad
        
        # Step size selection
        if gamma_selection == "optimal_line_search":
            gamma_k = optimal_line_search(f.f_numeric, xk, dk, interval)
        elif gamma_selection == "armijo":
            gamma_k = armijo_line_search(f.f_numeric, xk, dk, grad, gamma, alpha, beta)
        else:  # constant
            gamma_k = gamma
        
        # Update iterate
        xk = xk + gamma_k * dk
        
        # Compute new gradient
        grad = f.grad(xk)

        # Store results
        trajectory.append(xk.copy())
        grad_norms.append(np.linalg.norm(grad))
        f_vals.append(float(f(xk)))
        
        k += 1
    
    # Convert trajectory to array
    trajectory_array = np.array(trajectory)
    
    return trajectory_array, k, np.array(grad_norms), np.array(f_vals)
