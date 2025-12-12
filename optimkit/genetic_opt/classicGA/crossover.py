import numpy as np

def crossover(
    population_size: int,
    parents: np.ndarray,
    p_crossover: float,
    num_variables: int
) -> np.ndarray:
    """
    Perform single-point crossover on parent pairs.
    
    Args:
        population_size: Size of the population
        parents: Parent population array of shape (population_size, num_variables)
        p_crossover: Probability of crossover occurring
        num_variables: Number of variables (dimensions)
        
    Returns:
        Array of children of shape (population_size, num_variables)
    """
    children = np.zeros((population_size, num_variables))
    
    for i in range(0, population_size, 2):
        child1 = parents[i].copy()
        child2 = parents[i + 1].copy()
        
        # Perform crossover with probability p_crossover
        if np.random.rand() < p_crossover:
            # Randomly choose crossover point
            crossover_point = np.random.randint(1, num_variables)
            
            # Perform single-point crossover
            child1 = np.concatenate([
                parents[i][:crossover_point],
                parents[i + 1][crossover_point:]
            ])
            child2 = np.concatenate([
                parents[i + 1][:crossover_point],
                parents[i][crossover_point:]
            ])
        
        children[i] = child1
        children[i + 1] = child2
    
    return children