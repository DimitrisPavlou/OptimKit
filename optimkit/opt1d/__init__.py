"""
One-dimensional optimization methods.

This module provides various algorithms for finding the minimum of 
univariate functions.

Available Methods:
------------------
- golden_sector: Golden section search
- fibonacci: Fibonacci search  
- bisection: Bisection method
- diff_bisection: Differential bisection method
"""

from .golden_sector import golden_sector
from .fibonacci import fibonacci
from .bisection import bisection
from .diff_bisection import diff_bisection

__all__ = [
    "golden_sector",
    "fibonacci", 
    "bisection",
    "diff_bisection",
]