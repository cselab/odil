import statistics

import numpy as np
import scipy

# Van der Pol oscillator:  dx/dt = y,  dy/dt = μ(1-x²)y - x


def u(x, y):
    return y


def v(x, y):
    return mu * (1 - x**2) * y - x


def du_dx(x, y):
    return 0


def du_dy(x, y):
    return 1


def dv_dx(x, y):
    return -2 * mu * x * y - 1


def dv_dy(x, y):
    return mu * (1 - x**2)


def run_euler():
    x0, y0 = xi, yi
    s = [(x0, y0)]

    for _ in range(n):
        dx = dt * u(x0, y0)
        dy = dt * v(x0, y0)
        x0 += dx
        y0 += dy
        s.append((x0, y0))

    return s


def run_newton():
    k = 1 / dt
    x0, y0 = xi, yi
    s = [(x0, y0)]

    for _ in range(n):
        x1, y1 = x0, y0

        for __ in range(iter_newton):
            rx = (x1 - x0) * k - u(x1, y1)
            ry = (y1 - y0) * k - v(x1, y1)

            Jxx = k - du_dx(x1, y1)
            Jxy = -du_dy(x1, y1)
            Jyx = -dv_dx(x1, y1)
            Jyy = k - dv_dy(x1, y1)

            dx, dy = solve_2x2(Jxx, Jxy, Jyx, Jyy, rx, ry)
            x1 -= dx
            y1 -= dy

        x0, y0 = x1, y1
        s.append((x0, y0))

    return s


def run_odil_naive():
    X = [xi] + [xi] * n
    Y = [yi] + [yi] * n

    for _ in range(iter_naive):
        gx, gy = grad_loss(X, Y)
        for i in range(1, n + 1):
            X[i] -= lr_naive * gx[i]
            Y[i] -= lr_naive * gy[i]

    return list(zip(X, Y))


def run_odil_jacobian():
    X = [xi] + [xi] * n
    Y = [yi] + [yi] * n

    for _ in range(iter_odil):
        R = residual(X, Y)
        J = jacobian_dense(X, Y)
        dXY = linsolve(J, R)

        for i in range(n):
            X[i + 1] -= dXY[2 * i]
            Y[i + 1] -= dXY[2 * i + 1]

    return list(zip(X, Y))


def run_odil_sparse():
    X = [xi] + [xi] * n
    Y = [yi] + [yi] * n

    for _ in range(iter_odil):
        R = residual(X, Y)
        J = jacobian_sparse(X, Y)
        dXY = scipy.sparse.linalg.spsolve(J, R)

        for i in range(n):
            X[i + 1] -= dXY[2 * i]
            Y[i + 1] -= dXY[2 * i + 1]

    return list(zip(X, Y))


def solve_2x2(a, b, c, d, e, f):
    det = a * d - b * c
    x = (e * d - b * f) / det
    y = (a * f - e * c) / det
    return x, y


def residual(X, Y):
    R = []
    for i in range(1, n + 1):
        rx = (X[i] - X[i - 1]) / dt - u(X[i], Y[i])
        ry = (Y[i] - Y[i - 1]) / dt - v(X[i], Y[i])
        R.extend([rx, ry])
    return R


def jacobian_dense(X, Y):
    k = 1 / dt
    J = [[0] * (2 * n) for _ in range(2 * n)]

    for i in range(n):
        ix = 2 * i
        iy = 2 * i + 1

        if i > 0:
            J[ix][2 * i - 2] = -k
            J[iy][2 * i - 1] = -k

        J[ix][2 * i] = k - du_dx(X[i + 1], Y[i + 1])
        J[ix][2 * i + 1] = -du_dy(X[i + 1], Y[i + 1])

        J[iy][2 * i] = -dv_dx(X[i + 1], Y[i + 1])
        J[iy][2 * i + 1] = k - dv_dy(X[i + 1], Y[i + 1])

    return J


def jacobian_sparse(X, Y):
    k = 1 / dt
    row = []
    col = []
    dat = []

    for i in range(n):
        if i > 0:
            row.extend([2 * i, 2 * i + 1])
            col.extend([2 * i - 2, 2 * i - 1])
            dat.extend([-k, -k])

        x, y = X[i + 1], Y[i + 1]
        row.extend([2 * i, 2 * i])
        col.extend([2 * i, 2 * i + 1])
        dat.extend([k - du_dx(x, y), -du_dy(x, y)])

        row.extend([2 * i + 1, 2 * i + 1])
        col.extend([2 * i, 2 * i + 1])
        dat.extend([-dv_dx(x, y), k - dv_dy(x, y)])

    return scipy.sparse.csr_matrix((dat, (row, col)), dtype=float)


def linsolve(J, R):
    return np.linalg.solve(J, R).tolist()


def grad_loss(X, Y):
    k = 1 / dt
    gx = [0] * (n + 1)
    gy = [0] * (n + 1)

    for i in range(1, n + 1):
        rx = (X[i] - X[i - 1]) * k - u(X[i], Y[i])
        ry = (Y[i] - Y[i - 1]) * k - v(X[i], Y[i])

        gx[i] += 2 * rx * (k - du_dx(X[i], Y[i]))
        gy[i] += 2 * rx * (-du_dy(X[i], Y[i]))

        gx[i] += 2 * ry * (-dv_dx(X[i], Y[i]))
        gy[i] += 2 * ry * (k - dv_dy(X[i], Y[i]))

        if i > 1:
            gx[i - 1] -= 2 * rx * k
            gy[i - 1] -= 2 * ry * k

    return gx, gy


n = 25
dt = 0.1
xi = 2
yi = 0
mu = 2

iter_newton = 10
iter_odil = 10
iter_naive = 100000
lr_naive = 1e-3

for name, s in [
    ("Euler", run_euler()),
    ("Newton", run_newton()),
    ("ODIL naive", run_odil_naive()),
    ("ODIL jacobian", run_odil_jacobian()),
    ("ODIL sparse", run_odil_sparse()),
]:
    X, Y = zip(*s)
    res = statistics.fmean(residual(X, Y))
    print(f"{name:<14}  x: {s[-1][0]:6.3f}  y: {s[-1][1]:6.3f}  res: {res:.2e}")
