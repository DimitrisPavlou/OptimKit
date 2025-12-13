import numpy as np 
import sympy as sp 
from typing import Union, Callable, List, Literal
from numpy.typing import NDArray


class Function: 
    """
    A wrapper class for mathematical functions that handles both symbolic and numeric representations.
    
    This class provides a unified interface for working with functions, their gradients, and Hessians,
    supporting both univariate and multivariate cases.
    
    Attributes
    ----------
    func : Union[sp.Expr, Callable]
        The original function (symbolic expression or numeric callable).
    func_type : Literal["symbolic", "numeric"]
        Type of the function representation.
    n_vars : int
        Number of variables in the function.
    vars : List[sp.Symbol]
        Sorted list of symbolic variables (symbolic functions only).
    f_numeric : Callable
        Numeric callable version of the function.
    df_numeric : Callable
        Numeric callable for the derivative (univariate symbolic only).
    grad_f_numeric : Callable
        Numeric callable for the gradient (multivariate symbolic only).
    hessian_f_numeric : Callable
        Numeric callable for the Hessian (multivariate symbolic only).
    """
    
    def __init__(
        self, 
        func: Union[sp.Expr, Callable], 
        func_type: Literal["symbolic", "numeric"], 
        n_vars: int
    ) -> None:
        """
        Initialize a Function object.
        
        Parameters
        ----------
        func : Union[sp.Expr, Callable]
            Symbolic expression (SymPy) or numeric callable.
        func_type : Literal["symbolic", "numeric"]
            Type of function: "symbolic" for SymPy expressions, "numeric" for callables.
        n_vars : int
            Number of variables in the function.
            
        Raises
        ------
        ValueError
            If func_type is not "symbolic" or "numeric".
        """
        self.func: Union[sp.Expr, Callable] = func
        self.func_type: Literal["symbolic", "numeric"] = func_type
        self.n_vars: int = n_vars
       
        if self.func_type == "symbolic": 
            self.vars: List[sp.Symbol] = sorted(self.func.free_symbols, key=str)
            self.f_numeric: Callable = sp.lambdify(self.vars, self.func, "numpy")
            
            if self.n_vars == 1: 
                df_sym: sp.Expr = sp.diff(self.func, self.vars[0]) 
                self.df_numeric: Callable = sp.lambdify(self.vars, df_sym, 'numpy')
        
            if self.n_vars > 1: 
                grad_expr: sp.Matrix = sp.Matrix([self.func]).jacobian(self.vars)
                self.grad_f_numeric: Callable = sp.lambdify(self.vars, grad_expr, "numpy")
                hessian_expr: sp.Matrix = sp.hessian(self.func, self.vars)
                self.hessian_f_numeric: Callable = sp.lambdify(self.vars, hessian_expr, "numpy")
                
        elif self.func_type == "numeric": 
            self.f_numeric: Callable = self.func 
        else: 
            raise ValueError("Provide correct function type [symbolic, numeric]")
    
    def __call__(self, x: Union[float, NDArray[np.floating]]) -> float:
        """
        Evaluate the function at point x.
        
        Parameters
        ----------
        x : Union[float, NDArray[np.floating]]
            Point at which to evaluate. Scalar for univariate, array for multivariate.
            
        Returns
        -------
        float
            Function value at x.
            
        Notes
        -----
        - Numeric functions: Called with array as-is: f(x)
        - Symbolic functions: Arguments unpacked for multivariate: f(*x)
        """
        if self.func_type == "numeric":
            # Numeric callables take the array directly
            return float(self.f_numeric(x))
        else:
            # Symbolic functions need unpacking
            if self.n_vars == 1:
                return float(self.f_numeric(x))
            else:
                return float(self.f_numeric(*x))
    
    def grad(self, x: Union[float, NDArray[np.floating]]) -> Union[float, NDArray[np.floating]]:
        """
        Compute the gradient (or derivative) at point x.
        
        Parameters
        ----------
        x : Union[float, NDArray[np.floating]]
            Point at which to evaluate. Scalar for univariate, array for multivariate.
            
        Returns
        -------
        Union[float, NDArray[np.floating]]
            Derivative (scalar) for univariate, gradient array for multivariate.
            
        Raises
        ------
        AttributeError
            If gradient is not available (numeric function type).
        """
        if self.n_vars == 1: 
            return float(self.df_numeric(x))
        if self.n_vars > 1: 
            result = self.grad_f_numeric(*x)  
            return np.array(result, dtype=np.float64).flatten()
        
    def hessian(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Compute the Hessian matrix at point x.
        
        Parameters
        ----------
        x : NDArray[np.floating]
            Point at which to evaluate (multivariate).
            
        Returns
        -------
        NDArray[np.floating]
            Hessian matrix at x.
            
        Raises
        ------
        AttributeError
            If Hessian is not available (univariate or numeric function type).
        """
        result = self.hessian_f_numeric(*x)  
        return np.array(result, dtype=np.float64)