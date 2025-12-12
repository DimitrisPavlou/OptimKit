import numpy as np
from sympy import Expr, lambdify, Matrix, hessian
from typing import Callable, Union, Tuple, List
from scipy.linalg import eigh
from .helper_utils import optimal_line_search, armijo_line_search

NumericFunction = Callable[[np.ndarray], float]

def levenberg_marquardt(
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
    Levenberg-Marquardt method for multivariable function optimization.
    
    This method modifies the Hessian by adding a multiple of the identity matrix
    to ensure positive definiteness: Ak = Hk + mu*I, where mu is chosen based on
    the maximum eigenvalue of Hk.
    
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
        - x_min: Array of iterates (shape: [n_iterations, n_variables])
        - N: Number of iterations performed
        - grad_norms: Array of gradient norms at each iteration
        - f_vals: Array of function values at each iteration
        
    Raises:
        ValueError: If invalid gamma_selection method or missing required parameters
    """
    # Extract variables and create numeric functions
    vars_list = sorted(f.free_symbols, key=str)
    f_numeric = lambdify(vars_list, f, "numpy")
    
    # Compute gradient and Hessian symbolically
    grad_expr = Matrix([f]).jacobian(vars_list)
    hessian_expr = hessian(f, vars_list)
    
    grad_f_numeric = lambdify(vars_list, grad_expr, "numpy")
    hessian_f_numeric = lambdify(vars_list, hessian_expr, "numpy")
    
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
        # Compute Hessian at current point
        Hk = np.array(hessian_f_numeric(*xk), dtype=float)
        
        # Find the maximum absolute eigenvalue of Hk
        eigenvalues = np.linalg.eigvals(Hk)
        max_eigenvalue = np.max(np.abs(eigenvalues))
        
        # Compute mu to make Ak positive definite
        mu = max_eigenvalue + 1.0
        
        # Modified Hessian: Ak = Hk + mu*I
        Ak = Hk + mu * np.eye(n_vars)
        
        # Solve Ak * dk = -grad for the search direction
        try:
            dk = np.linalg.solve(Ak, -grad)
        except np.linalg.LinAlgError:
            print(f"Warning: Singular matrix at iteration {k}. Stopping.")
            break
        
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
