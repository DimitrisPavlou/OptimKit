"""
Grey Wolf Optimizer (GWO) for continuous optimization problems.

This module implements the Grey Wolf Optimizer algorithm, a nature-inspired 
metaheuristic algorithm that mimics the leadership hierarchy and hunting 
mechanism of grey wolves in nature.

Reference:
    Mirjalili, S., Mirjalili, S. M., & Lewis, A. (2014). Grey wolf optimizer. 
    Advances in engineering software, 69, 46-61.
"""

import numpy as np
from typing import Callable, Tuple, Union, List
from optimkit.function import Function
# Type alias
ObjectiveFunction = Callable[[np.ndarray], float]

def grey_wolf_optimizer(
    objective_function: Union[ObjectiveFunction, Function],
    lb: Union[np.ndarray, List[float]],
    ub: Union[np.ndarray, List[float]],
    dim: int,
    num_agents: int = 30,
    max_iterations: int = 500,
    print_every: int = 100
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Grey Wolf Optimizer for global optimization.
    
    The GWO algorithm simulates the leadership hierarchy and hunting behavior
    of grey wolves. The algorithm maintains three best solutions (alpha, beta, 
    delta) and updates the positions of search agents (omega wolves) based on 
    these leaders.
    
    This is a derivative-free metaheuristic method that works directly with
    function evaluations, making it suitable for black-box optimization problems.
    
    Args:
        objective_function: Either a raw callable that takes array and returns float,
                          or a Function object (can be "numeric" or "symbolic" type).
                          For metaheuristics, "numeric" Function objects are recommended
                          when you don't need gradients but want API consistency.
        lb: Lower bounds for each dimension
        ub: Upper bounds for each dimension
        dim: Number of dimensions
        num_agents: Number of search agents (wolves) (default: 30)
        max_iterations: Maximum number of iterations (default: 500)
        print_every: Print progress every N iterations (default: 100)
        
    Returns:
        Tuple containing:
        - best_fitness: Best fitness value found (Alpha score)
        - best_position: Best position found (Alpha position)
        - convergence_curve: Array of best fitness at each iteration
        
    Raises:
        ValueError: If bounds dimensions don't match dim or lb >= ub
        
    Examples:
        >>> # Option 1: Raw callable
        >>> def rosenbrock(x):
        ...     return 100*(x[1]-x[0]**2)**2 + (1-x[0])**2
        >>> best_fit, best_pos, curve = grey_wolf_optimizer(
        ...     rosenbrock, lb=[-5,-5], ub=[5,5], dim=2
        ... )
        
        >>> # Option 2: Function object (numeric)
        >>> from Function import Function
        >>> f = Function(rosenbrock, "numeric", 2)
        >>> best_fit, best_pos, curve = grey_wolf_optimizer(
        ...     f, lb=[-5,-5], ub=[5,5], dim=2
        ... )
    """
    # Input validation
    lb = np.array(lb) if not isinstance(lb, np.ndarray) else lb
    ub = np.array(ub) if not isinstance(ub, np.ndarray) else ub
    
    if len(lb) != dim or len(ub) != dim:
        raise ValueError(
            f"Bounds dimensions ({len(lb)}, {len(ub)}) must match dim ({dim})"
        )
    
    if np.any(lb >= ub):
        raise ValueError("Lower bounds must be strictly less than upper bounds")
    
    # Initialize the positions of the three best wolves (alpha, beta, delta)
    alpha_position = np.zeros(dim)
    alpha_score = float("inf")
    
    beta_position = np.zeros(dim)
    beta_score = float("inf")
    
    delta_position = np.zeros(dim)
    delta_score = float("inf")
    
    # Initialize the positions of search agents uniformly in the search space
    positions = np.random.uniform(
        low=lb, high=ub, size=(num_agents, dim)
    )
    
    # Storage for convergence tracking
    convergence_curve = np.zeros(max_iterations)
    
    # Main optimization loop
    for iteration in range(max_iterations):
        # Evaluate all search agents and update alpha, beta, delta
        for i in range(num_agents):
            # Clip positions to stay within bounds
            positions[i, :] = np.clip(positions[i, :], lb, ub)
            
            # Calculate objective function for current search agent
            fitness = objective_function(positions[i, :])
            
            # Update alpha (best solution)
            if fitness < alpha_score:
                delta_score = beta_score
                delta_position = beta_position.copy()
                beta_score = alpha_score
                beta_position = alpha_position.copy()
                alpha_score = fitness
                alpha_position = positions[i, :].copy()
            
            # Update beta (second best solution)
            elif fitness < beta_score:
                delta_score = beta_score
                delta_position = beta_position.copy()
                beta_score = fitness
                beta_position = positions[i, :].copy()
            
            # Update delta (third best solution)
            elif fitness < delta_score:
                delta_score = fitness
                delta_position = positions[i, :].copy()
        
        # Linearly decrease a from 2 to 0 over iterations
        a = 2.0 - iteration * (2.0 / max_iterations)
        
        # Update positions of all search agents
        for i in range(num_agents):
            for j in range(dim):
                # Generate random vectors for exploration/exploitation
                r1 = np.random.random()
                r2 = np.random.random()
                
                # Calculate parameters for alpha wolf
                A1 = 2.0 * a * r1 - a
                C1 = 2.0 * r2
                D_alpha = abs(C1 * alpha_position[j] - positions[i, j])
                X1 = alpha_position[j] - A1 * D_alpha
                
                # Calculate parameters for beta wolf
                r1 = np.random.random()
                r2 = np.random.random()
                A2 = 2.0 * a * r1 - a
                C2 = 2.0 * r2
                D_beta = abs(C2 * beta_position[j] - positions[i, j])
                X2 = beta_position[j] - A2 * D_beta
                
                # Calculate parameters for delta wolf
                r1 = np.random.random()
                r2 = np.random.random()
                A3 = 2.0 * a * r1 - a
                C3 = 2.0 * r2
                D_delta = abs(C3 * delta_position[j] - positions[i, j])
                X3 = delta_position[j] - A3 * D_delta
                
                # Update position based on alpha, beta, and delta
                positions[i, j] = (X1 + X2 + X3) / 3.0
        
        # Store best fitness for convergence tracking
        convergence_curve[iteration] = alpha_score
        
        # Print progress
        if iteration % print_every == 0:
            print(
                f"Iteration: {iteration:4d} | "
                f"Best Fitness: {alpha_score:.6e} | "
                f"Alpha Position: {alpha_position}"
            )
    
    # Final progress report
    print(
        f"\nOptimization Complete!\n"
        f"Best Fitness: {alpha_score:.6e}\n"
        f"Best Position: {alpha_position}"
    )
    
    return alpha_score, alpha_position, convergence_curve