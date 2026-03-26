"""
Comprehensive comparison of optimization methods on benchmark functions.

This script compares gradient-based methods (Steepest Descent, Newton, 
Levenberg-Marquardt), genetic algorithms, and Grey Wolf Optimizer on 
the Powell and Rosenbrock (Banana) functions.
"""

import sympy as sp
import numpy as np
from matplotlib import pyplot as plt
# Import ND gradient-based methods
from optimkit.optNd import newton_method, steepest_descent, levenberg_marquardt
# Import genetic algorithms and metaheuristics
from optimkit.genetic_opt import GA, grey_wolf_optimizer
from optimkit.function import Function

def powell(x: np.ndarray) -> float:
    """Powell function - a multimodal optimization test function."""
    return (x[0] + 10*x[1])**2 + 5*(x[2] - x[3])**2 + (x[1] - 2*x[2])**4 + 10*(x[0] - x[3])**4


def compare_optimization_methods():
    """
    Compare all optimization methods on the Powell function.
    """
    print("=" * 80)
    print("OPTIMIZATION METHODS COMPARISON - POWELL FUNCTION")
    print("=" * 80)
    
    # Define symbolic Powell function for gradient-based methods
    x1, x2, x3, x4 = sp.symbols('x1 x2 x3 x4')
    powell_sym = (x1 + 10*x2)**2 + 5*(x3 - x4)**2 + (x2 - 2*x3)**4 + 10*(x1 - x4)**4
    F_powel_sym = Function(powell_sym, "symbolic", n_vars=4)
    F_powel_num = Function(powell, "numeric", n_vars=4)
    
    # Starting point for all methods
    starting_point = np.array([3.0, -1.0, 0.0, 1.0])
    
    # Store results
    results = {}
    
    # =========================================================================
    # 1. STEEPEST DESCENT METHOD
    # =========================================================================
    print("\n" + "-" * 80)
    print("1. STEEPEST DESCENT METHOD (Armijo Line Search)")
    print("-" * 80)
    
    try:
        trajectory_sd, n_iter_sd, grad_norms_sd, f_vals_sd = steepest_descent(
            f=F_powel_sym,
            starting_point=starting_point,
            epsilon=1e-6,
            gamma_selection="armijo",
            gamma=1.0,
            alpha=1e-4,
            beta=0.5,
            max_iter=5000
        )
        
        results['Steepest Descent'] = {
            'final_point': trajectory_sd[-1],
            'final_value': f_vals_sd[-1],
            'iterations': n_iter_sd,
            'convergence': f_vals_sd
        }
        
        print(f"Converged in {n_iter_sd} iterations")
        print(f"Final point: {trajectory_sd[-1]}")
        print(f"Final function value: {f_vals_sd[-1]:.8e}")
        print(f"Final gradient norm: {grad_norms_sd[-1]:.8e}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        results['Steepest Descent'] = None
    
    # =========================================================================
    # 2. NEWTON'S METHOD
    # =========================================================================
    print("\n" + "-" * 80)
    print("2. NEWTON'S METHOD (Armijo Line Search)")
    print("-" * 80)
    
    try:
        trajectory_newton, n_iter_newton, grad_norms_newton, f_vals_newton = newton_method(
            f=F_powel_sym,
            starting_point=starting_point,
            epsilon=1e-6,
            gamma_selection="armijo",
            gamma=1.0,
            alpha=1e-4,
            beta=0.5,
            max_iter=5000
        )
        
        results['Newton'] = {
            'final_point': trajectory_newton[-1],
            'final_value': f_vals_newton[-1],
            'iterations': n_iter_newton,
            'convergence': f_vals_newton
        }
        
        print(f"Converged in {n_iter_newton} iterations")
        print(f"Final point: {trajectory_newton[-1]}")
        print(f"Final function value: {f_vals_newton[-1]:.8e}")
        print(f"Final gradient norm: {grad_norms_newton[-1]:.8e}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        results['Newton'] = None
    
    # =========================================================================
    # 3. LEVENBERG-MARQUARDT METHOD
    # =========================================================================
    print("\n" + "-" * 80)
    print("3. LEVENBERG-MARQUARDT METHOD (Armijo Line Search)")
    print("-" * 80)
    
    try:
        trajectory_lm, n_iter_lm, grad_norms_lm, f_vals_lm = levenberg_marquardt(
            f=F_powel_sym,
            starting_point=starting_point,
            epsilon=1e-6,
            gamma_selection="armijo",
            gamma=1.0,
            alpha=1e-4,
            beta=0.5,
            max_iter=5000
        )
        
        results['Levenberg-Marquardt'] = {
            'final_point': trajectory_lm[-1],
            'final_value': f_vals_lm[-1],
            'iterations': n_iter_lm,
            'convergence': f_vals_lm
        }
        
        print(f"Converged in {n_iter_lm} iterations")
        print(f"Final point: {trajectory_lm[-1]}")
        print(f"Final function value: {f_vals_lm[-1]:.8e}")
        print(f"Final gradient norm: {grad_norms_lm[-1]:.8e}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        results['Levenberg-Marquardt'] = None
    
    # =========================================================================
    # 4. GENETIC ALGORITHM
    # =========================================================================
    print("\n" + "-" * 80)
    print("4. GENETIC ALGORITHM (Tournament Selection)")
    print("-" * 80)
    
    try:
        best_solution_ga, best_fitness_ga, convergence_ga = GA(
            objective_function=F_powel_num,
            init_point=starting_point,
            population_size=100,
            max_generations=500,
            p_crossover=0.8,
            mutation_rate=0.01,
            selection_algorithm="tournament",
            num_elites=2,
            tournament_size=10,
            print_every=100
        )
        
        results['Genetic Algorithm'] = {
            'final_point': best_solution_ga,
            'final_value': best_fitness_ga,
            'iterations': 500,
            'convergence': convergence_ga
        }
        
        print(f"\nFinal Results:")
        print(f"Best solution: {best_solution_ga}")
        print(f"Best fitness: {best_fitness_ga:.8e}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        results['Genetic Algorithm'] = None
    
    # =========================================================================
    # 5. GREY WOLF OPTIMIZER
    # =========================================================================
    print("\n" + "-" * 80)
    print("5. GREY WOLF OPTIMIZER")
    print("-" * 80)
    
    # Define bounds for GWO
    dim = 4
    lb = -10 * np.ones(dim)
    ub = 10 * np.ones(dim)
    
    try:
        best_fitness_gwo, best_position_gwo, convergence_gwo = grey_wolf_optimizer(
            objective_function=F_powel_num,
            lb=lb,
            ub=ub,
            dim=dim,
            num_agents=30,
            max_iterations=500,
            print_every=100
        )
        
        results['Grey Wolf Optimizer'] = {
            'final_point': best_position_gwo,
            'final_value': best_fitness_gwo,
            'iterations': 500,
            'convergence': convergence_gwo
        }
        
    except Exception as e:
        print(f"ERROR: {e}")
        results['Grey Wolf Optimizer'] = None
    
    # =========================================================================
    # SUMMARY TABLE
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY OF RESULTS")
    print("=" * 80)
    print(f"{'Method':<25} {'Final Value':<20} {'Iterations':<15}")
    print("-" * 80)
    
    for method_name, result in results.items():
        if result is not None:
            print(f"{method_name:<25} {result['final_value']:<20.8e} {result['iterations']:<15}")
        else:
            print(f"{method_name:<25} {'FAILED':<20} {'-':<15}")
    
    # =========================================================================
    # CONVERGENCE PLOTS
    # =========================================================================
    print("\n" + "=" * 80)
    print("GENERATING CONVERGENCE PLOTS")
    print("=" * 80)
    
    plot_convergence(results)
    
    return results


def plot_convergence(results: dict):
    """
    Plot convergence curves for all successful methods.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Linear scale
    for method_name, result in results.items():
        if result is not None and 'convergence' in result:
            convergence = result['convergence']
            ax1.plot(convergence, label=method_name, linewidth=2)
    
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Function Value', fontsize=12)
    ax1.set_title('Convergence Comparison - Linear Scale', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Log scale
    for method_name, result in results.items():
        if result is not None and 'convergence' in result:
            convergence = result['convergence']
            # Filter out zero or negative values for log scale
            convergence_filtered = np.maximum(convergence, 1e-10)
            ax2.semilogy(convergence_filtered, label=method_name, linewidth=2)
    
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Function Value (log scale)', fontsize=12)
    ax2.set_title('Convergence Comparison - Log Scale', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig('powell_optimization_comparison.png', dpi=300, bbox_inches='tight')
    print("Convergence plots saved as 'powell_optimization_comparison.png'")
    plt.show()


if __name__ == "__main__":
    # Compare methods on Powell function
    powell_results = compare_optimization_methods()
    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)