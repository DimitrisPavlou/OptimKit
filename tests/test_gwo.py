"""Tests for Grey Wolf Optimizer."""
import numpy as np
import pytest
from optimkit.genetic_opt.GWO.GWO import grey_wolf_optimizer
from optimkit.function.Function import Function


class TestGreyWolfOptimizer:
    """Test Grey Wolf Optimizer."""

    def test_gwo_sphere_function(self):
        """Test GWO on sphere function."""
        np.random.seed(42)
        
        def sphere(x):
            return np.sum(x**2)
        
        best_fit, best_pos, curve = grey_wolf_optimizer(
            sphere,
            lb=[-10.0, -10.0],
            ub=[10.0, 10.0],
            dim=2,
            num_agents=10,
            max_iterations=50,
            print_every=100  # Disable printing during test
        )
        
        assert best_fit >= 0  # Sphere minimum is 0
        assert best_pos.shape == (2,)
        assert len(curve) == 50
        # Should converge toward origin
        assert np.linalg.norm(best_pos) < 1.0

    def test_gwo_rosenbrock_function(self):
        """Test GWO on Rosenbrock function."""
        np.random.seed(42)
        
        def rosenbrock(x):
            return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2
        
        best_fit, best_pos, curve = grey_wolf_optimizer(
            rosenbrock,
            lb=[-5.0, -5.0],
            ub=[5.0, 5.0],
            dim=2,
            num_agents=20,
            max_iterations=100,
            print_every=100
        )
        
        assert best_fit >= 0
        assert best_pos.shape == (2,)
        assert len(curve) == 100
        # Should converge toward (1, 1)
        assert np.linalg.norm(best_pos - np.array([1.0, 1.0])) < 2.0

    def test_gwo_with_function_object(self):
        """Test GWO with Function object."""
        np.random.seed(42)
        
        def sphere(x):
            return np.sum(x**2)
        
        f = Function(sphere, "numeric", n_vars=2)
        
        best_fit, best_pos, curve = grey_wolf_optimizer(
            f,
            lb=[-10.0, -10.0],
            ub=[10.0, 10.0],
            dim=2,
            num_agents=10,
            max_iterations=30,
            print_every=100
        )
        
        assert best_fit >= 0
        assert best_pos.shape == (2,)
        assert len(curve) == 30

    def test_gwo_invalid_num_agents(self):
        """Test that num_agents < 3 raises ValueError."""
        def sphere(x):
            return np.sum(x**2)
        
        with pytest.raises(ValueError, match="num_agents.*must be at least 3"):
            grey_wolf_optimizer(
                sphere,
                lb=[-10.0, -10.0],
                ub=[10.0, 10.0],
                dim=2,
                num_agents=2
            )

    def test_gwo_bounds_dimension_mismatch(self):
        """Test that bounds dimension mismatch raises ValueError."""
        def sphere(x):
            return np.sum(x**2)
        
        with pytest.raises(ValueError, match="Bounds dimensions.*must match dim"):
            grey_wolf_optimizer(
                sphere,
                lb=[-10.0],  # Wrong dimension
                ub=[10.0],
                dim=2
            )

    def test_gwo_invalid_bounds(self):
        """Test that lb >= ub raises ValueError."""
        def sphere(x):
            return np.sum(x**2)
        
        with pytest.raises(ValueError, match="Lower bounds must be strictly less"):
            grey_wolf_optimizer(
                sphere,
                lb=[10.0, 10.0],
                ub=[5.0, 5.0],  # Lower > upper
                dim=2
            )

    def test_gwo_convergence(self):
        """Test that GWO converges (best fitness improves)."""
        np.random.seed(42)
        
        def sphere(x):
            return np.sum(x**2)
        
        best_fit, best_pos, curve = grey_wolf_optimizer(
            sphere,
            lb=[-10.0, -10.0],
            ub=[10.0, 10.0],
            dim=2,
            num_agents=20,
            max_iterations=100,
            print_every=100
        )
        
        # Final fitness should be better than initial
        assert curve[-1] <= curve[0]
        # Curve should generally decrease (allow some fluctuation)
        assert curve[-1] < curve[0] * 0.5

    def test_gwo_3d_sphere(self):
        """Test GWO on 3D sphere function."""
        np.random.seed(42)
        
        def sphere(x):
            return np.sum(x**2)
        
        best_fit, best_pos, curve = grey_wolf_optimizer(
            sphere,
            lb=[-10.0, -10.0, -10.0],
            ub=[10.0, 10.0, 10.0],
            dim=3,
            num_agents=20,
            max_iterations=50,
            print_every=100
        )
        
        assert best_fit >= 0
        assert best_pos.shape == (3,)
        assert len(curve) == 50
        # Should converge toward origin
        assert np.linalg.norm(best_pos) < 2.0

    def test_gwo_minimum_agents(self):
        """Test GWO with minimum allowed agents (3)."""
        np.random.seed(42)
        
        def sphere(x):
            return np.sum(x**2)
        
        best_fit, best_pos, curve = grey_wolf_optimizer(
            sphere,
            lb=[-10.0, -10.0],
            ub=[10.0, 10.0],
            dim=2,
            num_agents=3,  # Minimum allowed
            max_iterations=20,
            print_every=100
        )
        
        assert best_fit >= 0
        assert best_pos.shape == (2,)
        assert len(curve) == 20
