import numpy as np

def mutation(
    children: np.ndarray,
    population_size: int,
    num_variables: int,
    mutation_rate: float,
    generation: int,
    max_generations: int,
    num_elites: int = 0
) -> None:
    """
    Apply Gaussian mutation to children with adaptive variance decay.
    
    The mutation variance decreases over generations to balance exploration
    and exploitation. This function modifies the children array in-place.
    
    Args:
        children: Children population array of shape (population_size, num_variables)
        population_size: Size of the population
        num_variables: Number of variables (dimensions)
        mutation_rate: Probability of mutation for each gene
        generation: Current generation number
        max_generations: Total number of generations
        num_elites: Number of elite individuals to preserve from mutation (default: 0)
    """
    # Adaptive variance decay
    initial_variance = 3.0
    variance_decay = initial_variance * (1.0 - generation / max_generations)
    variance_decay = max(variance_decay, 0.7)  # Minimum variance threshold
    
    # Apply mutation to non-elite individuals
    for i in range(num_elites, population_size):
        for j in range(num_variables):
            # Mutate with probability mutation_rate
            if np.random.rand() < mutation_rate:
                # Add Gaussian noise
                children[i, j] += np.random.normal(
                    loc=0.0, scale=np.sqrt(variance_decay)
                )

