import numpy as np

def elitism(
    population: np.ndarray,
    fitness: np.ndarray,
    num_elites: int
) -> np.ndarray:
    """
    Select the best individuals from the population (elitism).
    
    Args:
        population: Current population array of shape (population_size, num_variables)
        fitness: Fitness values for each individual of shape (population_size,)
        num_elites: Number of elite individuals to preserve
        
    Returns:
        Array of elite individuals of shape (num_elites, num_variables)
    """
    sorted_indices = np.argsort(fitness)
    elite_indices = sorted_indices[:num_elites]
    elites = population[elite_indices]
    return elites


def tournament_selection(
    population: np.ndarray,
    population_size: int,
    fitness: np.ndarray,
    num_elites: int = 0,
    k: int = 10
) -> np.ndarray:
    """
    Select parents using tournament selection.
    
    In tournament selection, k individuals are randomly chosen and the best
    one is selected as a parent. This process is repeated to fill the parent pool.
    
    Args:
        population: Current population array of shape (population_size, num_variables)
        population_size: Size of the population
        fitness: Fitness values for each individual of shape (population_size,)
        num_elites: Number of elite individuals to preserve (default: 0)
        k: Tournament size (default: 10)
        
    Returns:
        Array of selected parents of shape (population_size, num_variables)
    """
    num_variables = population.shape[1]
    parents = np.empty((population_size, num_variables))
    
    # Apply elitism if specified
    if num_elites > 0:
        elites = elitism(population, fitness, num_elites)
        parents[:num_elites] = elites
        num_parents = population_size - num_elites
    else:
        num_parents = population_size
    
    # Select the remaining parents using tournament selection
    for i in range(num_parents):
        # Randomly choose k individuals
        tournament_indices = np.random.choice(
            population_size, size=k, replace=False
        )
        # Find the best of the k individuals (minimum fitness)
        best_index = tournament_indices[np.argmin(fitness[tournament_indices])]
        # Add to parents array
        parents[num_elites + i] = population[best_index]
    
    return parents


def roulette_wheel_selection(
    population: np.ndarray,
    population_size: int,
    fitness: np.ndarray,
    num_elites: int = 0
) -> np.ndarray:
    """
    Select parents using roulette wheel selection.
    
    In roulette wheel selection, individuals are selected with probability
    proportional to their fitness. For minimization problems, fitness values
    are inverted.
    
    Args:
        population: Current population array of shape (population_size, num_variables)
        population_size: Size of the population
        fitness: Fitness values for each individual of shape (population_size,)
        num_elites: Number of elite individuals to preserve (default: 0)
        
    Returns:
        Array of selected parents of shape (population_size, num_variables)
    """
    # Transform fitness values for minimization problem
    # Add 1 to avoid division by zero
    fitness_transformed = 1.0 / (1.0 + fitness)
    
    # Calculate selection probabilities
    probabilities = fitness_transformed / np.sum(fitness_transformed)
    
    parents = []
    
    # Apply elitism if specified
    if num_elites > 0:
        elites = elitism(population, fitness, num_elites)
        parents.extend(elites)
    
    # Select the remaining parents using roulette wheel selection
    num_remaining = population_size - num_elites
    for i in range(0, num_remaining, 2):
        # Sample from the population based on the calculated probabilities
        parent_indices = np.random.choice(
            population_size, size=2, replace=False, p=probabilities
        )
        # Add selected parents
        parents.extend(population[parent_indices])
    
    return np.array(parents)
