# Optimizing a Discrete Loss (ODIL) Framework for forward problem

This tutorial demonstrates the ODIL framework for numerically integrating two-dimensional nonlinear ODE systems, implemented in [`compare.py`](compare.py). For this forward problem, ODIL is equivalent to traditional Newton-Raphson methods, but requires sparse matrix structure for efficiency.

## System
Van der Pol oscillator:
```
dx/dt = y
dy/dt = μ(1 - x²)y - x
```
Parameters: `xi = 2`, `yi = 0` (initial conditions), `μ = 2`, `n = 25` time steps, `dt = 0.1`.

## Methods

### 1. Explicit Euler
```
x_{n+1} = x_n + dt * u(x_n, y_n)
y_{n+1} = y_n + dt * v(x_n, y_n)
```

### 2. Step-wise Newton-Raphson
The non-linear implicit Euler system is:
```
x_{n+1} - x_n = dt * u(x_{n+1}, y_{n+1})
y_{n+1} - y_n = dt * v(x_{n+1}, y_{n+1})
```

Newton-Raphson solves `F(x,y) = 0` where F are the equations
```
F = [(x - x0)/dt - u(x,y), (y - y0)/dt - v(x,y)]  // 2 equations per step
J = ∂F/∂[x,y]
```

Iteration: `[x,y]^{(m+1)} = [x,y]^{(m)} - J^{-1} * F([x,y]^{(m)})` (m = iteration number)

### 3. ODIL Methods

In general ODIL optimizes discrete loss with data, but here we uses root finding since the system is determined.

Residuals are the same in step-wise Newton-Raphson:
```
r_{x,i} = (x_i - x_{i-1})/dt - u(x_i, y_i) = 0  // 50 residuals total (25 steps × 2 vars)
r_{y,i} = (y_i - y_{i-1})/dt - v(x_i, y_i) = 0
```
Loss: `L = Σ(r_{x,i}² + r_{y,i}²)`

Three ODIL approaches:

ODIL Naive (`run_odil_naive`): Gradient descent on loss
```
[X,Y]^{(m+1)} = [X,Y]^{(m)} - η * ∇L([X,Y]^{(m)})  // m = iteration number
```

ODIL Jacobian (`run_odil_jacobian`): Newton-Raphson on residuals with dense matrix solve
```
Solve: J * d[X,Y] = -r([X,Y]^{(m)})
[X,Y]^{(m+1)} = [X,Y]^{(m)} + d[X,Y]
```
where J is Jacobian of residuals r

ODIL Sparse (`run_odil_sparse`): Newton-Raphson on residuals with sparse matrix solve.

- ODIL Jacobian mirrors step-wise Newton but operates on full trajectory simultaneously
- Both exploit Jacobian sparsity (block-tridiagonal structure)