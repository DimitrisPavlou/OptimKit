# OptimKit

**optimkit** is a lightweight Python toolkit for classical numerical optimization algorithms, with an emphasis on clarity, mathematical correctness, and educational value.
It provides clean NumPy- and SymPy-based implementations of **one-dimensional**, **multivariable**, and **population-based** optimization methods.

The project is designed to closely follow standard optimization theory while remaining practical and easy to extend.

---

## ✨ Features

* 📐 **One-dimensional optimization**

  * Bisection
  * Derivative-based bisection
  * Fibonacci search
  * Golden-section search

* 📈 **Multivariable unconstrained optimization**

  * Steepest Descent
  * Newton’s Method
  * Levenberg–Marquardt
  * Multiple step-size (γ) selection strategies

* 🧬 **Metaheuristics / population-based methods**

  * Classical Genetic Algorithm (selection, crossover, mutation)
  * Grey Wolf Optimizer (GWO)

* 🧠 **Symbolic → Numeric workflow**

  * Objective functions defined symbolically using **SymPy**
  * Automatically converted to fast numeric functions via **NumPy**

* 🧩 Modular, readable, and easy to extend

---

## ⚠️ Problem Assumptions

At its current stage, **optimkit assumes unconstrained optimization problems**.

* No equality or inequality constraints are supported
* All multivariable methods operate on unconstrained objective functions
* Constraint handling (e.g. penalty methods, projections, Lagrange multipliers) is **not implemented yet**, but the project structure allows for future extensions.
  
Future versions may extend support to constrained optimization.

---

## 🔧 Installation

### Option 1: Install from source (recommended)

Clone the repository:

```bash
git clone https://github.com/your-username/optimkit.git
cd optimkit
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

### Option 2: Install with pip (future)

```bash
pip install optimkit
```

### Requirements

* Python ≥ 3.9
* NumPy ≥ 1.20.0
* SymPy ≥ 1.9.0

### Optional: Development setup

If you want to contribute or run the test suite:

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run the test suite
pytest tests/ -v

# Run tests with coverage report
pytest tests/ --cov=optimkit
```

---

## 🚀 Quick Usage Examples

### 1️⃣ One-dimensional optimization

```python
from sympy import symbols
from optimkit.function.Function import Function
from optimkit.opt1d import bisection

x = symbols('x')
f_expr = (x - 2)**2
f = Function(f_expr, "symbolic", n_vars=1)

low, high, ops = bisection(
    f=f,
    alpha=-5,
    beta=5,
    length_tol=1e-3,
    epsilon=1e-4
)

print("Approximate minimum:", (low[-1] + high[-1]) / 2)
```

---

### 2️⃣ Multivariable optimization (Steepest Descent)

```python
import sympy as sp
import numpy as np
from optimkit.function.Function import Function
from optimkit.optNd import steepest_descent

x, y = sp.symbols('x y')
f_expr = x**5 * sp.exp(-x**2 - y**2)
f = Function(f_expr, "symbolic", n_vars=2)

trajectory, n_iter, grad_norms, f_vals = steepest_descent(
    f=f,
    starting_point=np.array([-1.0, 1.0]),
    epsilon=1e-4,
    gamma_selection="constant",
    gamma=0.45,
    alpha=1e-4,
    beta=0.5
)

print("Minimum found at:", trajectory[-1])
```

---

### 3️⃣ Genetic Algorithm

```python
import numpy as np
from optimkit.genetic_opt.classicGA import GA

def sphere(x):
    return np.sum(x**2)

best_sol, best_fit, convergence = GA(
    objective_function=sphere,
    init_point=[5.0, 5.0],
    population_size=50,
    max_generations=100,
    p_crossover=0.7,
    mutation_rate=0.01,
    selection_algorithm="tournament",
    num_elites=2
)

print("Best solution:", best_sol)
print("Best fitness:", best_fit)
```

---

### 4️⃣ Grey Wolf Optimizer

```python
import numpy as np
from optimkit.genetic_opt.GWO.GWO import grey_wolf_optimizer

def sphere(x):
    return np.sum(x**2)

best_fit, best_pos, convergence = grey_wolf_optimizer(
    objective_function=sphere,
    lb=[-10.0, -10.0],
    ub=[10.0, 10.0],
    dim=2,
    num_agents=30,
    max_iterations=100
)

print("Best position:", best_pos)
print("Best fitness:", best_fit)
```

---

### 5️⃣ Step-size (γ) selection

Supported strategies for multivariable methods:

* **Constant step size**: `gamma_selection="constant"` (specify `gamma` parameter)
* **Armijo backtracking rule**: `gamma_selection="armijo"` (uses `alpha` and `beta` parameters)
* **Optimal line search**: `gamma_selection="optimal_line_search"` (uses golden-section search)

---

## 📁 Project Structure

```text
optimkit/
│
├── optimkit/           # Main package
│   ├── __init__.py
│   ├── function/       # Function wrapper class (Symbolic/Numeric)
│   │   └── Function.py
│   ├── opt1d/          # One-dimensional optimization algorithms
│   │   ├── bisection.py
│   │   ├── diff_bisection.py
│   │   ├── fibonacci.py
│   │   └── golden_sector.py
│   ├── optNd/          # Multivariable optimization algorithms
│   │   ├── steepest_descent.py
│   │   ├── newton.py
│   │   ├── levenberg_marquardt.py
│   │   └── helper_utils.py
│   └── genetic_opt/    # Population-based optimization methods
│       ├── classicGA/
│       │   ├── GA.py
│       │   ├── selection.py
│       │   ├── crossover.py
│       │   └── mutation.py
│       └── GWO/
│           └── GWO.py
│
├── tests/              # Test suite
│   ├── __init__.py
│   ├── test_function.py
│   ├── test_genetic.py
│   ├── test_gwo.py
│   ├── test_opt1d.py
│   └── test_optNd.py
│
├── requirements.txt
├── requirements-dev.txt
├── pytest.ini
└── README.md
```

---

## 🧠 Design Philosophy

* Prefer **clarity over excessive abstraction**
* Algorithms closely follow textbook formulations
* Symbolic definitions first, numeric execution second
* Minimal dependencies (NumPy + SymPy)
* Comprehensive test coverage
* Suitable for:

  * academic projects
  * numerical optimization coursework
  * research prototypes
  * learning optimization algorithms

---

## 🧪 Testing

The test suite covers all major functionality:

```bash
# Run all tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ --cov=optimkit

# Run specific test module
pytest tests/test_function.py -v
```

---

## 🛣️ Roadmap

Planned improvements:

* Constraint handling techniques (penalty methods, barrier methods)
* Quasi-Newton methods (BFGS, L-BFGS)
* Additional metaheuristics (PSO, Differential Evolution)
* Constrained optimization support
* Documentation generation (Sphinx)
* Performance benchmarks

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-method`)
3. Make your changes and add tests
4. Run the test suite (`pytest tests/ -v`)
5. Submit a pull request

---

## 📜 License

This project is released under the **MIT License**.

You are free to use, modify, and distribute it with attribution.

---

## 📚 Citation (optional)

If you use **optimkit** in academic work, consider citing it as a software artifact.
