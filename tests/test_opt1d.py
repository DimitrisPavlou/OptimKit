"""Tests for 1D optimization methods."""
import numpy as np
import sympy as sp
import pytest
from optimkit.opt1d.diff_bisection import diff_bisection
from optimkit.function.Function import Function


class TestDiffBisection:
    """Test derivative-based bisection method."""

    def test_diff_bisection_basic(self):
        """Test basic bisection on quadratic function."""
        x = sp.Symbol('x')
        f_expr = (x - 3)**2  # Minimum at x=3
        f = Function(f_expr, "symbolic", n_vars=1)
        
        a, b, n_ops = diff_bisection(f, alpha=0.0, beta=10.0, l=1e-5)
        
        # Check that we got some iterations
        assert n_ops > 0
        # Check that the final interval contains the minimum
        assert a[-1] <= 3.0 <= b[-1] or np.isclose(a[-1], 3.0, atol=0.1) or np.isclose(b[-1], 3.0, atol=0.1)

    def test_diff_bisection_early_termination(self):
        """Test bisection with early termination when derivative near zero."""
        x = sp.Symbol('x')
        f_expr = x**2  # Minimum at x=0
        f = Function(f_expr, "symbolic", n_vars=1)
        
        a, b, n_ops = diff_bisection(f, alpha=-5.0, beta=5.0, l=1e-5, tol=1e-6)
        
        # Should terminate early when derivative is near zero
        assert len(a) == n_ops
        assert len(b) == n_ops
        # Final interval should be around 0
        assert np.isclose((a[-1] + b[-1]) / 2, 0.0, atol=0.1)

    def test_diff_bisection_returns_correct_array_sizes(self):
        """Test that arrays are correctly sized on early termination."""
        x = sp.Symbol('x')
        f_expr = x**2
        f = Function(f_expr, "symbolic", n_vars=1)
        
        a, b, n_ops = diff_bisection(f, alpha=-1.0, beta=1.0, l=1e-5, tol=1e-6)
        
        # Arrays should have length equal to n_ops
        assert len(a) == n_ops
        assert len(b) == n_ops

    def test_diff_bisection_multivariate_raises_error(self):
        """Test that multivariate function raises ValueError."""
        x, y = sp.symbols('x y')
        f_expr = x**2 + y**2
        f = Function(f_expr, "symbolic", n_vars=2)
        
        with pytest.raises(ValueError, match="univariate"):
            diff_bisection(f, alpha=0.0, beta=10.0)

    def test_diff_bisection_numeric_raises_error(self):
        """Test that numeric function raises ValueError."""
        def quadratic(x):
            return x**2
        
        f = Function(quadratic, "numeric", n_vars=1)
        
        with pytest.raises(ValueError, match="symbolic"):
            diff_bisection(f, alpha=0.0, beta=10.0)

    def test_diff_bisection_cubic(self):
        """Test bisection on cubic function."""
        x = sp.Symbol('x')
        f_expr = x**3 - 3*x  # Local min at x=1
        f = Function(f_expr, "symbolic", n_vars=1)
        
        # df/dx = 3x^2 - 3, zero at x=1 and x=-1
        a, b, n_ops = diff_bisection(f, alpha=0.0, beta=5.0, l=1e-5)
        
        assert len(a) == n_ops
        assert len(b) == n_ops
        # Should converge to x=1
        assert np.isclose((a[-1] + b[-1]) / 2, 1.0, atol=0.1)

    def test_diff_bisection_precision(self):
        """Test bisection with different precision values."""
        x = sp.Symbol('x')
        f_expr = (x - 5)**2
        f = Function(f_expr, "symbolic", n_vars=1)
        
        # High precision
        a1, b1, n1 = diff_bisection(f, alpha=0.0, beta=10.0, l=1e-8)
        
        # Low precision
        a2, b2, n2 = diff_bisection(f, alpha=0.0, beta=10.0, l=1e-2)
        
        # Higher precision should require more iterations
        assert n1 >= n2
        # Final interval should be smaller with higher precision
        assert (b1[-1] - a1[-1]) <= (b2[-1] - a2[-1])
