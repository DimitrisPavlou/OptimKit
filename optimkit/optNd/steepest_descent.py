import numpy as np
from sympy import Expr, lambdify, symbols, Matrix
from typing import Callable, Union, Tuple, List
from .helper_utils import armijo_line_search, optimal_line_search

NumericFunction = Callable[[np.ndarray], float]

def steepest_descent(
    f: Expr,
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
        f: Symbolic expression of objective function (SymPy Expr)
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
    # Extract variables and create numeric functions
    # Sorting ensures consistent order when passing arguments to lambdified functions
    vars_list = sorted(f.free_symbols, key=str)
    f_numeric = lambdify(vars_list, f, "numpy")
    grad_expr = Matrix([f]).jacobian(vars_list)
    grad_f_numeric = lambdify(vars_list, grad_expr, "numpy")
    
    # Initialize
    xk = np.array(starting_point, dtype=float).flatten()
    n_vars = len(xk)
    
    # Storage for trajectory
    trajectory = [xk.copy()]
    grad = np.array(grad_f_numeric(*xk), dtype=float).flatten()
    grad_norms = [np.linalg.norm(grad)]
    f_vals = [float(f_numeric(*xk))]
    
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
            gamma_k = optimal_line_search(f_numeric, xk, dk, interval)
        elif gamma_selection == "armijo":
            gamma_k = armijo_line_search(f_numeric, xk, dk, grad, gamma, alpha, beta)
        else:  # constant
            gamma_k = gamma
        
        # Update iterate
        xk = xk + gamma_k * dk
        
        # Compute new gradient
        grad = np.array(grad_f_numeric(*xk), dtype=float).flatten()
        
        # Store results
        trajectory.append(xk.copy())
        grad_norms.append(np.linalg.norm(grad))
        f_vals.append(float(f_numeric(*xk)))
        
        k += 1
    
    # Convert trajectory to array
    trajectory_array = np.array(trajectory)
    
    return trajectory_array, k, np.array(grad_norms), np.array(f_vals)

