"""Tests for Genetic Algorithm components."""
import numpy as np
import pytest
from optimkit.genetic_opt.classicGA.crossover import crossover
from optimkit.genetic_opt.classicGA.mutation import mutation
from optimkit.genetic_opt.classicGA.selection import (
    elitism,
    tournament_selection,
    roulette_wheel_selection
)
from optimkit.genetic_opt.classicGA.GA import GA


class TestCrossover:
    """Test crossover function."""

    def test_crossover_single_variable(self):
        """Test crossover with single variable (num_variables=1)."""
        population_size = 4
        parents = np.array([[1.0], [2.0], [3.0], [4.0]])
        p_crossover = 0.7
        num_variables = 1
        
        children = crossover(population_size, parents, p_crossover, num_variables)
        
        assert children.shape == (4, 1)
        # With single variable, no actual crossover happens, just copy

    def test_crossover_odd_population_size(self):
        """Test crossover with odd population size."""
        population_size = 5
        parents = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])
        p_crossover = 0.7
        num_variables = 2
        
        children = crossover(population_size, parents, p_crossover, num_variables)
        
        assert children.shape == (5, 2)
        # Last individual should be copied without crossover

    def test_crossover_even_population_size(self):
        """Test crossover with even population size."""
        np.random.seed(42)
        population_size = 4
        parents = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        p_crossover = 1.0  # Always crossover
        num_variables = 2
        
        children = crossover(population_size, parents, p_crossover, num_variables)
        
        assert children.shape == (4, 2)

    def test_crossover_no_crossover_probability(self):
        """Test crossover with p_crossover=0 (no crossover)."""
        population_size = 4
        parents = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        p_crossover = 0.0
        num_variables = 2
        
        children = crossover(population_size, parents, p_crossover, num_variables)
        
        assert np.allclose(children, parents)


class TestMutation:
    """Test mutation function."""

    def test_mutation_basic(self):
        """Test basic mutation functionality."""
        np.random.seed(42)
        children = np.zeros((10, 2))
        population_size = 10
        num_variables = 2
        mutation_rate = 0.5  # High rate for testing
        generation = 0
        max_generations = 100
        
        mutation(children, population_size, num_variables, mutation_rate, 
                generation, max_generations)
        
        assert children.shape == (10, 2)

    def test_mutation_with_elites(self):
        """Test that elites are not mutated."""
        np.random.seed(42)
        children = np.ones((10, 2))
        population_size = 10
        num_variables = 2
        mutation_rate = 1.0  # Always mutate
        generation = 0
        max_generations = 100
        num_elites = 2
        
        mutation(children, population_size, num_variables, mutation_rate,
                generation, max_generations, num_elites)
        
        # First 2 (elites) should remain unchanged
        assert np.allclose(children[:2], 1.0)

    def test_mutation_invalid_num_elites(self):
        """Test that invalid num_elites raises ValueError."""
        children = np.zeros((10, 2))
        
        with pytest.raises(ValueError, match="num_elites.*must be less than"):
            mutation(children, population_size=10, num_variables=2,
                    mutation_rate=0.1, generation=0, max_generations=100,
                    num_elites=10)


class TestSelection:
    """Test selection functions."""

    def test_elitism_basic(self):
        """Test basic elitism selection."""
        population = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        fitness = np.array([5.0, 4.0, 3.0, 2.0, 1.0])  # Best is last
        num_elites = 2
        
        elites = elitism(population, fitness, num_elites)
        
        assert elites.shape == (2, 1)
        # Best two should be [5.0] and [4.0]
        assert np.allclose(elites[0], [5.0])
        assert np.allclose(elites[1], [4.0])

    def test_tournament_selection_basic(self):
        """Test basic tournament selection."""
        np.random.seed(42)
        population = np.array([[i] for i in range(10)])
        fitness = np.array([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
        population_size = 10
        
        parents = tournament_selection(population, population_size, fitness, k=3)
        
        assert parents.shape == (10, 1)

    def test_tournament_selection_invalid_k(self):
        """Test that k > population_size raises ValueError."""
        population = np.array([[1.0], [2.0], [3.0]])
        fitness = np.array([1.0, 2.0, 3.0])
        
        with pytest.raises(ValueError, match="Tournament size k.*cannot be greater"):
            tournament_selection(population, population_size=3, fitness=fitness, k=5)

    def test_roulette_wheel_selection_basic(self):
        """Test basic roulette wheel selection."""
        np.random.seed(42)
        population = np.array([[i] for i in range(10)])
        fitness = np.array([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
        population_size = 10
        
        parents = roulette_wheel_selection(population, population_size, fitness)
        
        assert parents.shape == (10, 1)

    def test_roulette_wheel_selection_with_elites(self):
        """Test roulette wheel selection with elites."""
        np.random.seed(42)
        population = np.array([[i] for i in range(10)])
        fitness = np.array([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
        population_size = 10
        num_elites = 2
        
        parents = roulette_wheel_selection(population, population_size, fitness, num_elites)
        
        assert parents.shape == (10, 1)


class TestGA:
    """Test full Genetic Algorithm."""

    def test_ga_sphere_function(self):
        """Test GA on sphere function."""
        np.random.seed(42)
        
        def sphere(x):
            return np.sum(x**2)
        
        best_sol, best_fit, curve = GA(
            sphere,
            init_point=[5.0, 5.0],
            population_size=50,
            max_generations=50,
            p_crossover=0.7,
            mutation_rate=0.1,
            tournament_size=10,  # Must be <= population_size
            print_every=100  # Disable printing during test
        )
        
        assert best_sol.shape == (2,)
        assert best_fit >= 0  # Sphere function minimum is 0
        assert len(curve) == 50
        # Check convergence (best fitness should decrease or stay same)
        assert np.all(np.diff(curve) <= 1e-10) or np.isclose(curve[-1], curve[0], atol=1e-6)

    def test_ga_with_function_object(self):
        """Test GA with Function object."""
        np.random.seed(42)
        from optimkit.function.Function import Function
        
        def sphere(x):
            return np.sum(x**2)
        
        f = Function(sphere, "numeric", n_vars=2)
        
        best_sol, best_fit, curve = GA(
            f,
            init_point=[5.0, 5.0],
            population_size=50,
            max_generations=30,
            tournament_size=10,  # Must be <= population_size
            print_every=100
        )
        
        assert best_sol.shape == (2,)
        assert len(curve) == 30

    def test_ga_roulette_selection(self):
        """Test GA with roulette wheel selection."""
        np.random.seed(42)
        
        def sphere(x):
            return np.sum(x**2)
        
        best_sol, best_fit, curve = GA(
            sphere,
            init_point=[5.0, 5.0],
            population_size=20,
            max_generations=30,
            selection_algorithm="roulette",
            print_every=100
        )
        
        assert best_sol.shape == (2,)
        assert len(curve) == 30

    def test_ga_invalid_selection_algorithm(self):
        """Test that invalid selection algorithm raises ValueError."""
        def sphere(x):
            return np.sum(x**2)
        
        with pytest.raises(ValueError, match="Invalid selection_algorithm"):
            GA(
                sphere,
                init_point=[5.0, 5.0],
                selection_algorithm="invalid"
            )
