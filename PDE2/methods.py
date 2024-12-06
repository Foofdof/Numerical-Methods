import numpy as np
from scipy.linalg import solve_banded


def explicit_scheme(funcs, init_cond, bound_cond, k, t_grid, x_grid):
    u = np.zeros((len(t_grid), len(x_grid)))
    alpha = (t_grid[1]-t_grid[0]) * k ** 2 / (x_grid[1]-x_grid[0]) ** 2

    u[0, ...] = init_cond[0](x_grid)
    u[..., 0] = bound_cond[0](t_grid)
    u[..., -1] = bound_cond[1](t_grid)

    for i in range(0, len(t_grid) - 1):
        u[i + 1, 1:-1] = (alpha * (u[i, 2:] + u[i, :-2]) + (1 - 2 * alpha) * u[i, 1:-1] +
                          (t_grid[1] - t_grid[0]) * funcs[0]([t_grid[i], x_grid[1:-1]]))

    return u


def explicit_scheme_ND(funcs, init_cond, bound_cond, k, t_grid, xy_grid):
    x, y = xy_grid
    nx = len(x)
    ny = len(y)
    nt = len(t_grid)

    h_x = x[1] - x[0]
    h_y = y[1] - y[0]
    tau = t_grid[1] - t_grid[0]
    stability_condition = tau <= (h_x**2 * h_y**2) / (2 * k * (h_x**2 + h_y**2))
    if not stability_condition:
        print("Stab")

    u = np.zeros((nt, nx, ny))

    X, Y = np.meshgrid(x, y, indexing='ij')
    u[0, :, :] = init_cond[0](X)

    u[0, 0, :] = bound_cond[0][0](t_grid[0])
    u[0, -1, :] = bound_cond[0][1](t_grid[0])
    u[0, :, 0] = bound_cond[1][0](t_grid[0])
    u[0, :, -1] = bound_cond[1][1](t_grid[0])

    for n in range(0, nt-1):
        u_old = u[n, :, :].copy()

        d2u_dx2 = (u_old[2:, 1:-1] - 2 * u_old[1:-1, 1:-1] + u_old[0:-2, 1:-1]) / h_x**2
        d2u_dy2 = (u_old[1:-1, 2:] - 2 * u_old[1:-1, 1:-1] + u_old[1:-1, 0:-2]) / h_y**2

        u[n+1, 1:-1, 1:-1] = u_old[1:-1, 1:-1] + k * tau * (d2u_dx2 + d2u_dy2) + tau * funcs[0](u_old[1:-1, 1:-1])

        u[n+1, 0, :] = bound_cond[0][0](t_grid[n+1])
        u[n+1, -1, :] = bound_cond[0][1](t_grid[n+1])
        u[n+1, :, 0] = bound_cond[1][0](t_grid[n+1])
        u[n+1, :, -1] = bound_cond[1][1](t_grid[n+1])

    return u


def implicit_scheme(funcs, init_cond, bound_cond, k, t_grid, x_grid):
    u = np.zeros((len(t_grid), len(x_grid)))
    alpha = (t_grid[1]-t_grid[0]) * k ** 2 / (x_grid[1]-x_grid[0]) ** 2

    u[0, ...] = init_cond[0](x_grid)
    u[..., 0] = bound_cond[0](t_grid)
    u[..., -1] = bound_cond[1](t_grid)

    A = np.diag((1 + 2 * alpha) * np.ones(len(x_grid)))
    A += np.diag(-alpha * np.ones(len(x_grid)-1), 1)
    A += np.diag(-alpha * np.ones(len(x_grid)-1), -1)

    A[0, :] = A[-1, :] = 0
    A[0, 0] = A[-1, -1] = 1

    for n in range(0, len(t_grid) - 1):
        b = np.zeros(len(A))
        b[1:-1] = u[n, 1:-1] + (t_grid[1] - t_grid[0]) * funcs[0]([t_grid[n], x_grid[1:-1]])

        b[0] = bound_cond[0](t_grid[n])
        b[-1] = bound_cond[1](t_grid[n])

        u[n + 1, 1:-1] = np.linalg.solve(A, b)[1:-1]

    return u


def weighted_impl_scheme(funcs, init_cond, bound_cond, k, t_grid, x_grid, sigma=1/2):
    u = np.zeros((len(t_grid), len(x_grid)))
    alpha = (t_grid[1]-t_grid[0]) * k ** 2 / (x_grid[1]-x_grid[0]) ** 2

    u[0, ...] = init_cond[0](x_grid)
    u[..., 0] = bound_cond[0](t_grid)
    u[..., -1] = bound_cond[1](t_grid)

    A = np.diag((1 + 2 * alpha * sigma) * np.ones(len(x_grid)))
    A += np.diag(-alpha * sigma * np.ones(len(x_grid)-1), 1)
    A += np.diag(-alpha * sigma * np.ones(len(x_grid)-1), -1)

    A[0, :] = A[-1, :] = 0
    A[0, 0] = A[-1, -1] = 1

    for n in range(0, len(t_grid) - 1):
        b = np.zeros(len(A))
        b[1: -1] = u[n, 1:-1] + (t_grid[1] - t_grid[0]) * (funcs[0]([t_grid[n], x_grid[1:-1]])+funcs[0]([t_grid[n+1], x_grid[1:-1]]))/2
        b[1:-1] += alpha * (1 - sigma) * (u[n, :-2] - 2 * u[n, 1:-1] + u[n, 2:])
        b[0] = bound_cond[0](t_grid[n])
        b[-1] = bound_cond[1](t_grid[n])

        u[n + 1, 1:-1] = np.linalg.solve(A, b)[1:-1]

    return u


def richardson(funcs, init_cond, bound_cond, k, t_grid, x_grid):
    u = np.zeros((len(t_grid), len(x_grid)))
    alpha = (t_grid[1]-t_grid[0]) * k ** 2 / (x_grid[1]-x_grid[0]) ** 2

    u[0, ...] = init_cond[0](x_grid)
    u[..., 0] = bound_cond[0](t_grid)
    u[..., -1] = bound_cond[1](t_grid)

    A = np.diag((1 + 2 * alpha) * np.ones(len(x_grid)))
    A += np.diag(-alpha * np.ones(len(x_grid)-1), 1)
    A += np.diag(-alpha * np.ones(len(x_grid)-1), -1)

    A[0, :] = A[-1, :] = 0
    A[0, 0] = A[-1, -1] = 1

    for n in range(0, 1):
        b = np.zeros(len(A))
        b[1:-1] = u[n, 1:-1] + (t_grid[1] - t_grid[0]) * funcs[0]([t_grid[n], x_grid[1:-1]])

        b[0] = bound_cond[0](t_grid[n])
        b[-1] = bound_cond[1](t_grid[n])

        u[n + 1, 1:-1] = np.linalg.solve(A, b)[1:-1]

    for n in range(1, len(t_grid) - 1):
        u[n + 1, 1:-1] = 2 * alpha * (u[n, :-2] - 2 * u[n, 1:-1] + u[n, 2:]) + 2*(t_grid[1] - t_grid[0]) * funcs[0]([t_grid[n], x_grid[1:-1]])
        u[n + 1, 1:-1] += u[n - 1, 1:-1]

    return u


def dufort_franc(funcs, init_cond, bound_cond, k, t_grid, x_grid):
    u = np.zeros((len(t_grid), len(x_grid)))
    alpha = (t_grid[1]-t_grid[0]) * k ** 2 / (x_grid[1]-x_grid[0]) ** 2

    u[0, ...] = init_cond[0](x_grid)
    u[..., 0] = bound_cond[0](t_grid)
    u[..., -1] = bound_cond[1](t_grid)

    A = np.diag((1 + 2 * alpha) * np.ones(len(x_grid)))
    A += np.diag(-alpha * np.ones(len(x_grid)-1), 1)
    A += np.diag(-alpha * np.ones(len(x_grid)-1), -1)

    A[0, :] = A[-1, :] = 0
    A[0, 0] = A[-1, -1] = 1

    for n in range(0, 1):
        b = np.zeros(len(A))
        b[1:-1] = u[n, 1:-1] + (t_grid[1] - t_grid[0]) * funcs[0]([t_grid[n], x_grid[1:-1]])

        b[0] = bound_cond[0](t_grid[n])
        b[-1] = bound_cond[1](t_grid[n])

        u[n + 1, 1:-1] = np.linalg.solve(A, b)[1:-1]

    for n in range(1, len(t_grid) - 1):
        u[n + 1, 1:-1] = 2 * alpha * (u[n, :-2] + u[n, 2:] - u[n - 1, 1:-1]) + 2*(t_grid[1] - t_grid[0]) * funcs[0]([t_grid[n], x_grid[1:-1]])
        u[n + 1, 1:-1] += u[n - 1, 1:-1]
        u[n + 1, 1:-1] /= (1+2*alpha)

    return u


def adi_solver(funcs2d, init_cond2d, bound_cond2d, k, t_grid, xy_grid):
    x, y = xy_grid

    Nx = len(x)
    Ny = len(y)
    tau = t_grid[1] - t_grid[0]

    snapshots = np.zeros((len(t_grid), Nx, Ny))

    X, Y = np.meshgrid(x, y, indexing='ij')
    U = init_cond2d[0](X)


    def apply_boundary(U, t):
        U[:, 0] = bound_cond2d[0][0](t)
        U[:, -1] = bound_cond2d[0][1](t)
        U[0, :] = bound_cond2d[1][0](t)
        U[-1, :] = bound_cond2d[1][1](t)
        return U

    U = apply_boundary(U, t_grid[0])

    h_x = x[1] - x[0]
    h_y = y[1] - y[0]

    alpha_x = k * tau / (2 * h_x ** 2)
    alpha_y = k * tau / (2 * h_y ** 2)

    def construct_tridiag_solver(n, alpha):
        main_diag = np.ones(n) * (1 + 2 * alpha)
        low_diag = -alpha * np.ones(n - 1)
        up_diag = -alpha * np.ones(n - 1)
        ab = np.zeros((3, n))
        ab[0, 1:] = up_diag
        ab[1, :] = main_diag
        ab[2, :-1] = low_diag
        return ab

    ab_x = construct_tridiag_solver(Nx - 2, alpha_x)
    ab_y = construct_tridiag_solver(Ny - 2, alpha_y)

    def D_xx(U):
        return (U[2:, :] - 2 * U[1:-1, :] + U[:-2, :])

    def D_yy(U):
        return (U[:, 2:] - 2 * U[:, 1:-1] + U[:, :-2])

    for n in range(len(t_grid)):
        t = t_grid[n]

        f_n = np.zeros_like(U)
        for f in funcs2d:
            f_n += f([t, X, Y])

        RHS_1 = U.copy()
        RHS_1[1:-1, 1:-1] += alpha_y * D_yy(U)[1:-1, :] + (tau / 2) * f_n[1:-1, 1:-1]
        for j in range(1, Ny - 1):
            rhs_line = RHS_1[1:-1, j].copy()
            rhs_line[0] += alpha_x * U[0, j]
            rhs_line[-1] += alpha_x * U[-1, j]
            U[1:-1, j] = solve_banded((1, 1), ab_x, rhs_line)

        U_half = U.copy()
        U_half = apply_boundary(U_half, t + tau / 2)

        f_half = np.zeros_like(U_half)
        for f in funcs2d:
            f_half += f([X, Y, t + tau / 2])

        RHS_2 = U_half.copy()
        RHS_2[1:-1, 1:-1] += alpha_x * D_xx(U_half)[:, 1:-1] + (tau / 2) * f_half[1:-1, 1:-1]

        for i in range(1, Nx - 1):
            rhs_line = RHS_2[i, 1:-1].copy()
            rhs_line[0] += alpha_y * U_half[i, 0]
            rhs_line[-1] += alpha_y * U_half[i, -1]
            U[i, 1:-1] = solve_banded((1, 1), ab_y, rhs_line)

        U = apply_boundary(U, t + tau)

        snapshots[n, :, :] = U.copy()

    return snapshots


def diff_conv_expl_scheme(funcs, init_cond, bound_cond, k, c, t_grid, x_grid):
    u = np.zeros((len(t_grid), len(x_grid)))
    alpha = (t_grid[1]-t_grid[0]) * k ** 2 / (x_grid[1]-x_grid[0]) ** 2
    beta = (t_grid[1] - t_grid[0]) * c / (x_grid[1] - x_grid[0]) / 2

    u[0, ...] = init_cond[0](x_grid)
    u[..., 0] = bound_cond[0](t_grid)
    u[..., -1] = bound_cond[1](t_grid)

    for i in range(0, len(t_grid) - 1):
        u[i + 1, 1:-1] = alpha * (u[i, 2:] + u[i, :-2]) + (1 - 2 * alpha) * u[i, 1:-1]
        u[i + 1, 1:-1] -= beta*(u[i, 2:] - u[i, :-2])
        u[i + 1, 1:-1] += (t_grid[1] - t_grid[0]) * funcs[0]([t_grid[i], x_grid[1:-1]])

    return u


def diff_conv_impl_scheme(funcs, init_cond, bound_cond, k, c, t_grid, x_grid):
    u = np.zeros((len(t_grid), len(x_grid)))
    alpha = (t_grid[1]-t_grid[0]) * k ** 2 / (x_grid[1]-x_grid[0]) ** 2
    beta = (t_grid[1] - t_grid[0]) * c / (x_grid[1] - x_grid[0]) / 2

    u[0, ...] = init_cond[0](x_grid)
    u[..., 0] = bound_cond[0](t_grid)
    u[..., -1] = bound_cond[1](t_grid)

    A = np.diag((1 + 2 * alpha + beta) * np.ones(len(x_grid)))
    A += np.diag(-alpha * np.ones(len(x_grid) - 1), 1)
    A += np.diag(-alpha * np.ones(len(x_grid) - 1), -1)

    A[0, :] = A[-1, :] = 0
    A[0, 0] = A[-1, -1] = 1

    for n in range(0, len(t_grid) - 1):
        b = np.zeros(len(A))
        b[1:-1] = u[n, 1:-1] + (t_grid[1] - t_grid[0]) * funcs[0]([t_grid[n], x_grid[1:-1]])
        b[1:-1] -= beta*(u[n, 2:] - u[n, :-2])

        b[0] = bound_cond[0](t_grid[n])
        b[-1] = bound_cond[1](t_grid[n])

        u[n + 1, 1:-1] = np.linalg.solve(A, b)[1:-1]

    return u
