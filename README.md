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

Clone the repository:

```bash
git clone https://github.com/your-username/optimkit.git
cd optimkit
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

### Requirements

* Python ≥ 3.9
* NumPy
* SymPy

---

## 🚀 Quick Usage Examples

### 1️⃣ One-dimensional optimization

```python
from sympy import symbols
from optimkit.opt1d import bisection

x = symbols('x')
f = (x - 2)**2

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
from optimkit.optNd import steepest_descent

x, y = sp.symbols('x y')
f = x**5 * sp.exp(-x**2 - y**2)

x_min, N, grad_norm, f_vals = steepest_descent(
    f=f,
    epsilon=1e-4,
    starting_point=np.array([-1.0, 1.0]),
    gamma_selection="constant",
    gamma=0.45,
    alpha=None,
    beta=None
)

print("Minimum found at:", x_min[-1])
```

---

### 3️⃣ Step-size (γ) selection

Supported strategies for multivariable methods:

* **Constant step size**
* **Armijo backtracking rule**
* **Optimal line search** (via line search on ( f(x_k + \gamma d_k) ))

---

## 📁 Project Structure

```text
optimkit/
│
├── opt1d/              # One-dimensional optimization algorithms
│   ├── bisection.py
│   ├── diff_bisection.py
│   ├── fibonacci.py
│   └── golden_sector.py
│
├── optNd/              # Multivariable optimization algorithms
│   ├── steepest_descent.py
│   ├── newton.py
│   ├── levenberg_marquardt.py
│   └── helper_utils.py
│
├── genetic_opt/        # Population-based optimization methods
│   ├── classicGA/
│   │   ├── GA.py
│   │   ├── selection.py
│   │   ├── crossover.py
│   │   └── mutation.py
│   └── GWO/
│       └── GWO.py
│
├── requirements.txt
└── README.md
```

---

## 🧠 Design Philosophy

* Prefer **clarity over excessive abstraction**
* Algorithms closely follow textbook formulations
* Symbolic definitions first, numeric execution second
* Minimal dependencies
* Suitable for:

  * academic projects
  * numerical optimization coursework
  * research prototypes

---

## 🛣️ Roadmap

Planned improvements:

* Constraint handling techniques
* Quasi-Newton methods (BFGS, L-BFGS)
* Additional metaheuristics
* Documentation generation (Sphinx)

---

## 📜 License

This project is released under the **MIT License**.

You are free to use, modify, and distribute it with attribution.

---

## 📚 Citation (optional)

If you use **optimkit** in academic work, consider citing it as a software artifact.
