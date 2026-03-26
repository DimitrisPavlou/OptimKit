"""Tests for N-dimensional optimization methods."""
import numpy as np
import sympy as sp
import pytest
from optimkit.optNd.steepest_descent import steepest_descent
from optimkit.optNd.newton import newton_method
from optimkit.optNd.levenberg_marquardt import levenberg_marquardt
from optimkit.function.Function import Function


class TestSteepestDescent:
    """Test steepest descent optimization."""

    def test_steepest_descent_sphere(self):
        """Test steepest descent on sphere function."""
        x, y = sp.symbols('x y')
        f_expr = x**2 + y**2
        f = Function(f_expr, "symbolic", n_vars=2)
        
        trajectory, n_iter, grad_norms, f_vals = steepest_descent(
            f, starting_point=[5.0, 5.0], epsilon=1e-6, max_iter=1000
        )
        
        assert n_iter > 0
        assert len(trajectory) == n_iter + 1
        assert len(grad_norms) == n_iter + 1
        assert len(f_vals) == n_iter + 1
        # Final point should be close to origin
        assert np.linalg.norm(trajectory[-1]) < 0.1
        # Final gradient norm should be small
        assert grad_norms[-1] < 1e-6

    def test_steepest_descent_rosenbrock(self):
        """Test steepest descent on Rosenbrock function."""
        x, y = sp.symbols('x y')
        f_expr = 100*(y - x**2)**2 + (1 - x)**2
        f = Function(f_expr, "symbolic", n_vars=2)
        
        trajectory, n_iter, grad_norms, f_vals = steepest_descent(
            f, starting_point=[0.0, 0.0], epsilon=1e-6, max_iter=5000
        )
        
        assert n_iter > 0
        # Should converge toward (1, 1)
        final_point = trajectory[-1]
        assert np.linalg.norm(final_point - np.array([1.0, 1.0])) < 1.0

    def test_steepest_descent_constant_step(self):
        """Test steepest descent with constant step size."""
        x, y = sp.symbols('x y')
        f_expr = x**2 + y**2
        f = Function(f_expr, "symbolic", n_vars=2)
        
        trajectory, n_iter, grad_norms, f_vals = steepest_descent(
            f, starting_point=[5.0, 5.0], epsilon=1e-6,
            gamma_selection="constant", gamma=0.1, max_iter=100
        )
        
        assert n_iter > 0
        assert grad_norms[-1] < 1e-6 or n_iter == 100

    def test_steepest_descent_univariate_raises_error(self):
        """Test that univariate function raises ValueError."""
        x = sp.Symbol('x')
        f_expr = x**2
        f = Function(f_expr, "symbolic", n_vars=1)
        
        with pytest.raises(ValueError, match="multivariate"):
            steepest_descent(f, starting_point=[5.0])

    def test_steepest_descent_numeric_raises_error(self):
        """Test that numeric function raises ValueError."""
        def sphere(x):
            return np.sum(x**2)
        
        f = Function(sphere, "numeric", n_vars=2)
        
        with pytest.raises(ValueError, match="symbolic"):
            steepest_descent(f, starting_point=[5.0, 5.0])

    def test_steepest_descent_invalid_gamma_selection(self):
        """Test that invalid gamma_selection raises ValueError."""
        x, y = sp.symbols('x y')
        f_expr = x**2 + y**2
        f = Function(f_expr, "symbolic", n_vars=2)
        
        with pytest.raises(ValueError, match="Unknown gamma_selection"):
            steepest_descent(f, starting_point=[5.0, 5.0], gamma_selection="invalid")

    def test_steepest_descent_dimension_mismatch(self):
        """Test that dimension mismatch raises ValueError."""
        x, y = sp.symbols('x y')
        f_expr = x**2 + y**2
        f = Function(f_expr, "symbolic", n_vars=2)
        
        with pytest.raises(ValueError, match="Starting point dimension"):
            steepest_descent(f, starting_point=[5.0])


class TestNewtonMethod:
    """Test Newton's method optimization."""

    def test_newton_sphere(self):
        """Test Newton's method on sphere function."""
        x, y = sp.symbols('x y')
        f_expr = x**2 + y**2
        f = Function(f_expr, "symbolic", n_vars=2)
        
        trajectory, n_iter, grad_norms, f_vals = newton_method(
            f, starting_point=[5.0, 5.0], epsilon=1e-6, max_iter=100
        )
        
        assert n_iter > 0
        # Newton should converge quickly for quadratic
        assert n_iter < 50
        # Final point should be close to origin
        assert np.linalg.norm(trajectory[-1]) < 0.1

    def test_newton_rosenbrock(self):
        """Test Newton's method on Rosenbrock function."""
        x, y = sp.symbols('x y')
        f_expr = 100*(y - x**2)**2 + (1 - x)**2
        f = Function(f_expr, "symbolic", n_vars=2)
        
        trajectory, n_iter, grad_norms, f_vals = newton_method(
            f, starting_point=[0.0, 0.0], epsilon=1e-6, max_iter=100
        )
        
        assert n_iter > 0
        # Should converge toward (1, 1)
        final_point = trajectory[-1]
        assert np.linalg.norm(final_point - np.array([1.0, 1.0])) < 1.0

    def test_newton_univariate_raises_error(self):
        """Test that univariate function raises ValueError."""
        x = sp.Symbol('x')
        f_expr = x**2
        f = Function(f_expr, "symbolic", n_vars=1)
        
        with pytest.raises(ValueError, match="multivariate"):
            newton_method(f, starting_point=[5.0])


class TestLevenbergMarquardt:
    """Test Levenberg-Marquardt optimization."""

    def test_lm_sphere(self):
        """Test LM on sphere function."""
        x, y = sp.symbols('x y')
        f_expr = x**2 + y**2
        f = Function(f_expr, "symbolic", n_vars=2)
        
        trajectory, n_iter, grad_norms, f_vals = levenberg_marquardt(
            f, starting_point=[5.0, 5.0], epsilon=1e-6, max_iter=100
        )
        
        assert n_iter > 0
        # Final point should be close to origin
        assert np.linalg.norm(trajectory[-1]) < 0.1

    def test_lm_rosenbrock(self):
        """Test LM on Rosenbrock function."""
        x, y = sp.symbols('x y')
        f_expr = 100*(y - x**2)**2 + (1 - x)**2
        f = Function(f_expr, "symbolic", n_vars=2)
        
        trajectory, n_iter, grad_norms, f_vals = levenberg_marquardt(
            f, starting_point=[0.0, 0.0], epsilon=1e-6, max_iter=100
        )
        
        assert n_iter > 0
        # Should converge toward (1, 1)
        final_point = trajectory[-1]
        assert np.linalg.norm(final_point - np.array([1.0, 1.0])) < 1.0

    def test_lm_univariate_raises_error(self):
        """Test that univariate function raises ValueError."""
        x = sp.Symbol('x')
        f_expr = x**2
        f = Function(f_expr, "symbolic", n_vars=1)
        
        with pytest.raises(ValueError, match="multivariate"):
            levenberg_marquardt(f, starting_point=[5.0])


class TestHelperUtils:
    """Test helper utilities for N-dimensional optimization."""

    def test_armijo_line_search(self):
        """Test Armijo line search."""
        from optimkit.optNd.helper_utils import armijo_line_search
        
        def sphere(x):
            return np.sum(x**2)
        
        xk = np.array([5.0, 5.0])
        dk = np.array([-1.0, -1.0])  # Descent direction
        grad = np.array([10.0, 10.0])
        
        gamma = armijo_line_search(sphere, xk, dk, grad)
        
        assert gamma > 0
        assert gamma <= 1.0

    def test_optimal_line_search(self):
        """Test optimal line search."""
        from optimkit.optNd.helper_utils import optimal_line_search
        
        def sphere(x):
            return np.sum(x**2)
        
        xk = np.array([5.0, 5.0])
        dk = np.array([-1.0, -1.0])
        
        gamma = optimal_line_search(sphere, xk, dk)
        
        assert gamma > 0
        # For sphere, optimal step should be around 0.5 in this direction
        assert 0.1 < gamma < 10.0

    def test_golden_section_search(self):
        """Test golden section search."""
        from optimkit.optNd.helper_utils import golden_section_search
        
        def quadratic(x):
            return (x - 3)**2
        
        result = golden_section_search(quadratic, a=0.0, b=10.0, tol=1e-6)
        
        assert np.isclose(result, 3.0, atol=1e-5)
