import numpy as np
from .selection import tournament_selection , roulette_wheel_selection
from .crossover import crossover
from .mutation import mutation
from typing import Callable, Tuple, Literal, Union, List
from optimkit.function import Function

ObjectiveFunction = Callable[[np.ndarray], float]
SelectionMethod = Literal["tournament", "roulette"]

def GA(
    objective_function: Union[ObjectiveFunction, Function],
    init_point: Union[np.ndarray, List[float]],
    population_size: int = 100,
    max_generations: int = 100,
    p_crossover: float = 0.7,
    mutation_rate: float = 0.001,
    selection_algorithm: SelectionMethod = "tournament",
    num_elites: int = 0,
    tournament_size: int = 30,
    print_every: int = 100
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Genetic Algorithm for continuous optimization.
    
    This is a derivative-free metaheuristic method that works directly with
    function evaluations, making it suitable for complex, non-differentiable,
    or black-box optimization problems.
    
    Args:
        objective_function: Either a raw callable that takes array and returns float,
                          or a Function object (can be "numeric" or "symbolic" type).
                          For metaheuristics, "numeric" Function objects are recommended
                          when you don't need gradients but want API consistency.
        init_point: Initial point to center the population around
        population_size: Number of individuals in population (default: 100)
        max_generations: Maximum number of generations (default: 100)
        p_crossover: Probability of crossover (default: 0.7)
        mutation_rate: Probability of mutation per gene (default: 0.001)
        selection_algorithm: Selection method - "tournament" or "roulette" (default: "tournament")
        num_elites: Number of elite individuals to preserve (default: 0)
        tournament_size: Size of tournament for tournament selection (default: 30)
        print_every: Print progress every N generations (default: 100)
        
    Returns:
        Tuple containing:
        - best_solution: Best solution found of shape (num_variables,)
        - best_fitness: Fitness value of the best solution
        - convergence_curve: Array of best fitness at each generation of shape (max_generations,)
        
    Raises:
        ValueError: If invalid selection_algorithm specified
        
    Examples:
        >>> # Option 1: Raw callable
        >>> def sphere(x):
        ...     return np.sum(x**2)
        >>> best_sol, best_fit, curve = GA(
        ...     sphere, init_point=[5, 5, 5], population_size=50
        ... )
        
        >>> # Option 2: Function object (numeric)
        >>> from Function import Function
        >>> f = Function(sphere, "numeric", 3)
        >>> best_sol, best_fit, curve = GA(
        ...     f, init_point=[5, 5, 5], population_size=50
        ... )
    """
    # Validate inputs
    if selection_algorithm not in ["tournament", "roulette"]:
        raise ValueError(
            f"Invalid selection_algorithm: '{selection_algorithm}'. "
            "Choose 'tournament' or 'roulette'."
        )
    
    if not isinstance(init_point, np.ndarray):
        init_point = np.array(init_point)
    
    # Number of variables (dimensions)
    num_variables = len(init_point)
    
    # Initialize population centered around init_point
    population = init_point * np.ones((population_size, num_variables))
    
    # Storage for convergence tracking
    convergence_curve = np.zeros(max_generations)
    
    # Best solution tracking
    best_solution = None
    best_fitness = float('inf')
    
    # Main evolutionary loop
    for generation in range(max_generations):
        # Evaluate fitness of each individual in the population
        fitness = np.array([objective_function(ind) for ind in population])
        
        # Select parents
        if selection_algorithm == "roulette":
            parents = roulette_wheel_selection(
                population, population_size, fitness, num_elites
            )
        else:  # tournament
            parents = tournament_selection(
                population, population_size, fitness, num_elites, k=tournament_size
            )
        
        # Create offspring through crossover
        children = crossover(population_size, parents, p_crossover, num_variables)
        
        # Apply mutation
        mutation(
            children, population_size, num_variables, 
            mutation_rate, generation, max_generations, num_elites
        )
        
        # Replace population with offspring
        population = children
        
        # Track best solution in current generation
        best_index = np.argmin(fitness)
        current_best_solution = population[best_index]
        current_best_fitness = fitness[best_index]
        
        # Update global best
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_solution = current_best_solution.copy()
        
        convergence_curve[generation] = best_fitness
        
        # Print progress
        if generation % print_every == 0:
            print(
                f"Generation: {generation:4d} | "
                f"Best Fitness: {best_fitness:.6e} | "
                f"Best Solution: {best_solution}"
            )
    
    # Final report
    print(
        f"\nOptimization Complete!\n"
        f"Best Fitness: {best_fitness:.6e}\n"
        f"Best Solution: {best_solution}"
    )
    
    return best_solution, best_fitness, convergence_curve