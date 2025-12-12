import numpy as np
from .selection import tournament_selection , roulette_wheel_selection
from .crossover import crossover
from .mutation import mutation
from typing import Callable, Tuple, Literal

ObjectiveFunction = Callable[[np.ndarray], float]
SelectionMethod = Literal["tournament", "roulette"]


def GA(
    objective_function: ObjectiveFunction,
    init_point: np.ndarray,
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
    
    Args:
        objective_function: Function to minimize, takes array and returns float
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
    
    return best_solution, best_fitness, convergence_curve