"""Tests for the Function class."""
import numpy as np
import sympy as sp
import pytest
from optimkit.function.Function import Function


class TestFunctionInit:
    """Test Function initialization."""

    def test_symbolic_function_creation(self):
        """Test creating a symbolic function."""
        x = sp.Symbol('x')
        f_expr = x**2 + 2*x + 1
        f = Function(f_expr, "symbolic", n_vars=1)
        
        assert f.func_type == "symbolic"
        assert f.n_vars == 1
        assert callable(f.f_numeric)

    def test_numeric_function_creation(self):
        """Test creating a numeric function."""
        def f_callable(x):
            return np.sum(x**2)
        
        f = Function(f_callable, "numeric", n_vars=2)
        
        assert f.func_type == "numeric"
        assert f.n_vars == 2
        assert callable(f.f_numeric)

    def test_invalid_func_type_raises_error(self):
        """Test that invalid func_type raises ValueError."""
        x = sp.Symbol('x')
        with pytest.raises(ValueError, match="symbolic.*numeric"):
            Function(x**2, "invalid", n_vars=1)


class TestFunctionCall:
    """Test Function __call__ method."""

    def test_symbolic_univariate_call(self):
        """Test calling a symbolic univariate function."""
        x = sp.Symbol('x')
        f_expr = x**2 + 2*x + 1
        f = Function(f_expr, "symbolic", n_vars=1)
        
        result = f(2.0)
        expected = 2.0**2 + 2*2.0 + 1  # 9.0
        assert np.isclose(result, expected)

    def test_symbolic_multivariate_call(self):
        """Test calling a symbolic multivariate function."""
        x, y = sp.symbols('x y')
        f_expr = x**2 + y**2
        f = Function(f_expr, "symbolic", n_vars=2)
        
        result = f(np.array([3.0, 4.0]))
        expected = 3.0**2 + 4.0**2  # 25.0
        assert np.isclose(result, expected)

    def test_numeric_function_call(self):
        """Test calling a numeric function."""
        def sphere(x):
            return np.sum(x**2)
        
        f = Function(sphere, "numeric", n_vars=2)
        result = f(np.array([3.0, 4.0]))
        assert np.isclose(result, 25.0)


class TestFunctionGrad:
    """Test Function grad method."""

    def test_symbolic_univariate_grad(self):
        """Test gradient of symbolic univariate function."""
        x = sp.Symbol('x')
        f_expr = x**3 + 2*x**2 + 1
        f = Function(f_expr, "symbolic", n_vars=1)
        
        # df/dx = 3x^2 + 4x
        result = f.grad(2.0)
        expected = 3*2.0**2 + 4*2.0  # 20.0
        assert np.isclose(result, expected)

    def test_symbolic_multivariate_grad(self):
        """Test gradient of symbolic multivariate function."""
        x, y = sp.symbols('x y')
        f_expr = x**2 + y**2
        f = Function(f_expr, "symbolic", n_vars=2)
        
        result = f.grad(np.array([3.0, 4.0]))
        expected = np.array([6.0, 8.0])
        assert np.allclose(result, expected)

    def test_numeric_function_grad_raises_error(self):
        """Test that grad raises AttributeError for numeric functions."""
        def sphere(x):
            return np.sum(x**2)
        
        f = Function(sphere, "numeric", n_vars=2)
        
        with pytest.raises(AttributeError, match="Gradient not available"):
            f.grad(np.array([1.0, 2.0]))


class TestFunctionHessian:
    """Test Function hessian method."""

    def test_symbolic_multivariate_hessian(self):
        """Test Hessian of symbolic multivariate function."""
        x, y = sp.symbols('x y')
        f_expr = x**2 + 2*x*y + y**2
        f = Function(f_expr, "symbolic", n_vars=2)
        
        result = f.hessian(np.array([1.0, 2.0]))
        expected = np.array([[2.0, 2.0], [2.0, 2.0]])
        assert np.allclose(result, expected)

    def test_univariate_hessian_raises_error(self):
        """Test that hessian raises AttributeError for univariate functions."""
        x = sp.Symbol('x')
        f_expr = x**3 + 2*x**2 + 1
        f = Function(f_expr, "symbolic", n_vars=1)
        
        with pytest.raises(AttributeError, match="Hessian not available"):
            f.hessian(np.array([2.0]))

    def test_numeric_function_hessian_raises_error(self):
        """Test that hessian raises AttributeError for numeric functions."""
        def sphere(x):
            return np.sum(x**2)
        
        f = Function(sphere, "numeric", n_vars=2)
        
        with pytest.raises(AttributeError, match="Hessian not available"):
            f.hessian(np.array([1.0, 2.0]))


class TestFunctionEdgeCases:
    """Test edge cases for Function class."""

    def test_rosenbrock_function(self):
        """Test with Rosenbrock function."""
        x, y = sp.symbols('x y')
        f_expr = 100*(y - x**2)**2 + (1 - x)**2
        f = Function(f_expr, "symbolic", n_vars=2)
        
        # At minimum (1, 1), function value should be 0
        result = f(np.array([1.0, 1.0]))
        assert np.isclose(result, 0.0, atol=1e-10)
        
        # Gradient at minimum should be zero
        grad = f.grad(np.array([1.0, 1.0]))
        assert np.allclose(grad, [0.0, 0.0], atol=1e-10)

    def test_sphere_function_3d(self):
        """Test with 3D sphere function."""
        x, y, z = sp.symbols('x y z')
        f_expr = x**2 + y**2 + z**2
        f = Function(f_expr, "symbolic", n_vars=3)
        
        result = f(np.array([1.0, 2.0, 3.0]))
        expected = 1.0 + 4.0 + 9.0  # 14.0
        assert np.isclose(result, expected)
        
        grad = f.grad(np.array([1.0, 2.0, 3.0]))
        expected_grad = np.array([2.0, 4.0, 6.0])
        assert np.allclose(grad, expected_grad)
