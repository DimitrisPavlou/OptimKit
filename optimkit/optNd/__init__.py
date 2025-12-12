"""
Multi-dimensional gradient-based optimization methods.

This module provides gradient-based algorithms for finding the minimum of 
multi-variable functions.

Available Methods:
------------------
- steepest_descent: Steepest descent with Armijo/optimal line search
- newton_method: Newton's method with Hessian
- levenberg_marquardt: Levenberg-Marquardt with modified Hessian
"""

from .steepest_descent import steepest_descent
from .newton import newton_method
from .levenberg_marquardt import levenberg_marquardt

__all__ = [
    "steepest_descent",
    "newton_method",
    "levenberg_marquardt",
]
