"""
OptimKit - A Comprehensive Optimization Toolkit
================================================

OptimKit provides a collection of optimization algorithms for continuous 
optimization problems, including gradient-based methods, genetic algorithms,
and swarm intelligence approaches.

Modules:
--------
- opt1d: One-dimensional optimization methods
- optNd: Multi-dimensional gradient-based optimization methods
- genetic_opt: Genetic algorithms and metaheuristic methods

Available Methods:
------------------

**One-Dimensional Optimization (optimkit.opt1d):**
    - golden_sector: Golden section search
    - fibonacci: Fibonacci search
    - bisection: Bisection method
    - diff_bisection: Differential bisection method

**Multi-Dimensional Gradient-Based Methods (optimkit.optNd):**
    - steepest_descent: Steepest descent with various line search strategies
    - newton_method: Newton's method with Hessian
    - levenberg_marquardt: Levenberg-Marquardt with modified Hessian

**Genetic Algorithms & Metaheuristics (optimkit.genetic_opt):**
    - GA: Genetic algorithm with tournament/roulette selection
    - grey_wolf_optimizer: Grey Wolf Optimizer (GWO)

Example Usage:
--------------
>>> import numpy as np
>>> from optimkit.optNd import steepest_descent, newton_method
>>> from optimkit.genetic_opt import GA, grey_wolf_optimizer
>>> from optimkit.opt1d import golden_sector, bisection
>>> from sympy import symbols
>>> 
>>> # Define a function symbolically
>>> x, y = symbols('x y')
>>> f = x**2 + y**2
>>> 
>>> # Optimize using Newton's method
>>> result = newton_method(f, starting_point=[1.0, 1.0])
>>> 
>>> # Optimize using genetic algorithm
>>> def objective(x):
...     return x[0]**2 + x[1]**2
>>> best_sol, best_fit, conv = GA(objective, init_point=np.array([1.0, 1.0]))
"""

__version__ = "1.0.0"
__author__ = "Your Name"

# Import submodules to make them accessible
from . import opt1d
from . import optNd
from . import genetic_opt

__all__ = [
    "opt1d",
    "optNd", 
    "genetic_opt",
]


def get_version():
    """Return the version of OptimKit."""
    return __version__


def list_methods():
    """
    Print all available optimization methods in OptimKit.
    """
    print("OptimKit - Available Optimization Methods")
    print("=" * 60)
    print("\n1D Optimization Methods (optimkit.opt1d):")
    print("  - golden_sector")
    print("  - fibonacci")
    print("  - bisection")
    print("  - diff_bisection")
    print("\nMulti-dimensional Gradient-Based Methods (optimkit.optNd):")
    print("  - steepest_descent")
    print("  - newton_method")
    print("  - levenberg_marquardt")
    print("\nGenetic Algorithms & Metaheuristics (optimkit.genetic_opt):")
    print("  - GA")
    print("  - grey_wolf_optimizer")
    print("=" * 60)
    print("\nExample imports:")
    print("  from optimkit.opt1d import golden_sector, bisection")
    print("  from optimkit.optNd import newton_method, steepest_descent")
    print("  from optimkit.genetic_opt import GA, grey_wolf_optimizer")
    print("=" * 60)