"""
Comprehensive comparison of optimization methods on benchmark functions.

This script compares gradient based/free methods (bisection, differential bisection, fibonacci, golden sector) on different functions to test their capability.
"""

import numpy as np
import sympy as sp
from optimkit.function.Function import Function
from optimkit.opt1d import bisection, diff_bisection, fibonacci, golden_sector

def test_bisection():
    """Test bisection method on f(x) = (x-2)^2"""
    print("=" * 60)
    print("Testing Bisection Method")
    print("=" * 60)
    
    x = sp.Symbol('x')
    f_sym = (x - 2)**2
    f = Function(f_sym, "symbolic", 1)
    
    print(f"Function: f(x) = {f_sym}")
    print(f"True minimum: x* = 2")
    print(f"Searching in interval [0, 5]")
    
    a, b, num_ops = bisection(f, 0.0, 5.0, length_tol=1e-5, epsilon=1e-6)
    
    final_midpoint = 0.5 * (a[-1] + b[-1])
    print(f"\nResults:")
    print(f"  Number of iterations: {len(a)}")
    print(f"  Number of function evaluations: {num_ops}")
    print(f"  Final interval: [{a[-1]:.8f}, {b[-1]:.8f}]")
    print(f"  Final midpoint: {final_midpoint:.8f}")
    print(f"  Error: {abs(final_midpoint - 2.0):.2e}")
    print(f"  f(x*): {f(final_midpoint):.2e}")
    
    assert abs(final_midpoint - 2.0) < 1e-4, "Bisection failed to find minimum"
    print("✓ Test passed!\n")


def test_diff_bisection():
    """Test derivative-based bisection method on f(x) = x^2 - 4x + 4"""
    print("=" * 60)
    print("Testing Derivative-Based Bisection Method")
    print("=" * 60)
    
    x = sp.Symbol('x')
    f_sym = x**2 - 4*x + 4
    f = Function(f_sym, "symbolic", 1)
    
    print(f"Function: f(x) = {f_sym}")
    print(f"Derivative: f'(x) = {sp.diff(f_sym, x)}")
    print(f"True minimum: x* = 2")
    print(f"Searching in interval [0, 5]")
    
    a, b, num_ops = diff_bisection(f, 0.0, 5.0, l=1e-5, tol=1e-8)
    
    final_midpoint = 0.5 * (a[-1] + b[-1])
    print(f"\nResults:")
    print(f"  Number of iterations: {len(a)}")
    print(f"  Number of derivative evaluations: {num_ops}")
    print(f"  Final interval: [{a[-1]:.8f}, {b[-1]:.8f}]")
    print(f"  Final midpoint: {final_midpoint:.8f}")
    print(f"  Error: {abs(final_midpoint - 2.0):.2e}")
    print(f"  f(x*): {f(final_midpoint):.2e}")
    print(f"  f'(x*): {f.grad(final_midpoint):.2e}")
    
    assert abs(final_midpoint - 2.0) < 1e-4, "Derivative bisection failed to find minimum"
    print("✓ Test passed!\n")


def test_fibonacci():
    """Test Fibonacci search on f(x) = x^3 - 3x + 1"""
    print("=" * 60)
    print("Testing Fibonacci Search Method")
    print("=" * 60)
    
    x = sp.Symbol('x')
    f_sym = x**3 - 3*x + 1
    f = Function(f_sym, "symbolic", 1)
    
    print(f"Function: f(x) = {f_sym}")
    print(f"Derivative: f'(x) = {sp.diff(f_sym, x)}")
    
    # True minimum is at x = 1 (where derivative = 3x^2 - 3 = 0)
    true_min = 1.0
    print(f"True minimum: x* ≈ {true_min}")
    print(f"Searching in interval [-2, 3]")
    
    a, b, n = fibonacci(f, -2.0, 3.0, length_tol=1e-5, epsilon=1e-6)
    
    final_midpoint = 0.5 * (a[-1] + b[-1])
    print(f"\nResults:")
    print(f"  Number of iterations: {len(a)}")
    print(f"  Final interval: [{a[-1]:.8f}, {b[-1]:.8f}]")
    print(f"  Final midpoint: {final_midpoint:.8f}")
    print(f"  Error: {abs(final_midpoint - true_min):.2e}")
    print(f"  f(x*): {f(final_midpoint):.6f}")
    
    assert abs(final_midpoint - true_min) < 1e-3, "Fibonacci search failed to find minimum"
    print("✓ Test passed!\n")


def test_golden_sector():
    """Test Golden Section search on f(x) = e^x - 2x"""
    print("=" * 60)
    print("Testing Golden Section Search Method")
    print("=" * 60)
    
    x = sp.Symbol('x')
    f_sym = sp.exp(x) - 2*x
    f = Function(f_sym, "symbolic", 1)
    
    print(f"Function: f(x) = {f_sym}")
    print(f"Derivative: f'(x) = {sp.diff(f_sym, x)}")
    
    # True minimum is at x = ln(2) ≈ 0.693
    true_min = np.log(2)
    print(f"True minimum: x* ≈ {true_min:.6f}")
    print(f"Searching in interval [-1, 2]")
    
    a, b, num_ops = golden_sector(f, -1.0, 2.0, length_tol=1e-5)
    
    final_midpoint = 0.5 * (a[-1] + b[-1])
    print(f"\nResults:")
    print(f"  Number of iterations: {len(a)}")
    print(f"  Number of function evaluations: {num_ops}")
    print(f"  Final interval: [{a[-1]:.8f}, {b[-1]:.8f}]")
    print(f"  Final midpoint: {final_midpoint:.8f}")
    print(f"  Error: {abs(final_midpoint - true_min):.2e}")
    print(f"  f(x*): {f(final_midpoint):.6f}")
    
    assert abs(final_midpoint - true_min) < 1e-4, "Golden section search failed to find minimum"
    print("✓ Test passed!\n")


def test_comparison():
    """Compare all methods on the same function"""
    print("=" * 60)
    print("Comparison of All Methods")
    print("=" * 60)
    
    x = sp.Symbol('x')
    f_sym = (x - np.pi)**2 + 0.5
    f = Function(f_sym, "symbolic", 1)
    
    true_min = np.pi
    interval = [0, 5]
    tol = 1e-6
    
    print(f"Function: f(x) = (x - π)^2 + 0.5")
    print(f"True minimum: x* = π ≈ {true_min:.8f}")
    print(f"Searching in interval {interval}")
    print(f"Tolerance: {tol}")
    print()
    
    results = {}
    
    # Bisection
    a, b, ops = bisection(f, interval[0], interval[1], tol, 1e-7)
    x_min = 0.5 * (a[-1] + b[-1])
    results['Bisection'] = {
        'x_min': x_min,
        'error': abs(x_min - true_min),
        'iterations': len(a),
        'f_evals': ops
    }
    
    # Derivative Bisection
    a, b, ops = diff_bisection(f, interval[0], interval[1], tol, 1e-8)
    x_min = 0.5 * (a[-1] + b[-1])
    results['Derivative Bisection'] = {
        'x_min': x_min,
        'error': abs(x_min - true_min),
        'iterations': len(a),
        'f_evals': ops
    }
    
    # Fibonacci
    a, b, n = fibonacci(f, interval[0], interval[1], tol, 1e-7)
    x_min = 0.5 * (a[-1] + b[-1])
    results['Fibonacci'] = {
        'x_min': x_min,
        'error': abs(x_min - true_min),
        'iterations': len(a),
        'f_evals': n
    }
    
    # Golden Section
    a, b, ops = golden_sector(f, interval[0], interval[1], tol)
    x_min = 0.5 * (a[-1] + b[-1])
    results['Golden Section'] = {
        'x_min': x_min,
        'error': abs(x_min - true_min),
        'iterations': len(a),
        'f_evals': ops
    }
    
    # Print comparison table
    print(f"{'Method':<25} {'x*':<12} {'Error':<12} {'Iters':<8} {'f_evals':<10}")
    print("-" * 75)
    for method, res in results.items():
        print(f"{method:<25} {res['x_min']:<12.8f} {res['error']:<12.2e} "
              f"{res['iterations']:<8} {res['f_evals']:<10}")
    print()
    
    # Find best method
    best_error = min(res['error'] for res in results.values())
    best_method = [m for m, res in results.items() if res['error'] == best_error][0]
    print(f"Best accuracy: {best_method} (error: {best_error:.2e})")
    
    fewest_evals = min(res['f_evals'] for res in results.values())
    most_efficient = [m for m, res in results.items() if res['f_evals'] == fewest_evals][0]
    print(f"Most efficient: {most_efficient} ({fewest_evals} function evaluations)")
    print()


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("1D OPTIMIZATION METHODS TEST SUITE")
    print("=" * 60 + "\n")
    
    try:
        test_bisection()
        test_diff_bisection()
        test_fibonacci()
        test_golden_sector()
        test_comparison()
        
        print("=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
