import numpy as np

def solve_fredholm_tr(k, f, l, grid):
    """
    :param k: kernum func
    :param f: right part of eq.
    :param l: lambda
    :param grid: grid
    :return: np.array
    """
    h = grid[1]-grid[0]

    grid_np = np.array(grid)
    k_matrix = k(grid_np[:, np.newaxis], grid_np[np.newaxis, :])
    k_matrix *= (l * h)

    k_matrix[:, 0] /= 2.
    k_matrix[:, -1] /= 2.

    k_matrix += np.eye(k_matrix.shape[1])
    f_vec = f(grid)

    f_vec = f_vec.reshape(-1, 1)
    sol = np.linalg.solve(k_matrix, f_vec)
    return sol


def solve_fredholm_leg(k, f, l, n, rng):
    """
    :param k: kernum func
    :param f: right part of eq.
    :param l: lambda
    :param n: num of legendre polynomial
    :param rng: (a, b)
    :return: np.array
    """
    nodes, weights = np.polynomial.legendre.leggauss(n)
    grid_np = (rng[1] - rng[0]) * nodes / 2 + (rng[1] + rng[0]) / 2
    k_matrix = k(grid_np[:, np.newaxis], grid_np[np.newaxis, :])
    k_matrix[:] *= weights
    k_matrix *= l
    k_matrix += np.eye(k_matrix.shape[1])
    f_vec = f(grid_np)

    f_vec = f_vec.reshape(-1, 1)
    sol = np.linalg.solve(k_matrix, f_vec)
    return sol, grid_np


# def solve_volterra_tr(k, f, l, grid):
#     """
#     :param k: kernum func
#     :param f: right part of eq.
#     :param l: lambda
#     :param grid: grid
#     :return: np.array
#     """
#     h = grid[1]-grid[0]
#
#     k_matrix = np.zeros((grid.shape[0], grid.shape[0]))
#
#     for i in range(grid.shape[0]):
#         for j in range(i):
#             k_matrix[i, j] += h/2*k(grid[i], grid[j])
#             k_matrix[i, j+1] += h/2*k(grid[i], grid[j+1])
#
#     k_matrix += np.eye(k_matrix.shape[1])
#     f_vec = f(grid)
#     f_vec = f_vec.reshape(-1, 1)
#     sol = np.linalg.solve(k_matrix, f_vec)
#     return sol


def solve_volterra_tr(k, f, l, grid):
    """
    :param k: kernum func
    :param f: right part of eq.
    :param l: lambda
    :param grid: grid
    :return: np.array
    """
    h = grid[1] - grid[0]

    grid_np = np.array(grid)
    k_matrix = k(grid_np[:, np.newaxis], grid_np[np.newaxis, :])
    k_matrix = np.tril(k_matrix)
    k_matrix *= (l * h)

    k_matrix[:, 0] /= 2.
    k_matrix[:, -1] /= 2.

    k_matrix += np.eye(k_matrix.shape[1])
    f_vec = f(grid)

    f_vec = f_vec.reshape(-1, 1)
    sol = np.linalg.solve(k_matrix, f_vec)
    return sol


# def solve_volterra_tr(k, f, l, grid):
#     """
#     :param k: kernum func
#     :param f: right part of eq.
#     :param l: lambda
#     :param grid: grid
#     :return: np.array
#     """
#     h = grid[1]-grid[0]
#
#     grid_np = np.array(grid)
#     i, j = np.tril_indices(grid.shape[0], -1)
#     k_matrix = np.zeros((grid.shape[0], grid.shape[0]))
#     k_matrix[i, j] = h / 2 * k(grid_np[i], grid_np[j])
#     k_matrix[i, j + 1] = h / 2 * k(grid_np[i], grid_np[j + 1])
#
#     k_matrix += np.eye(k_matrix.shape[1])
#     f_vec = f(grid)
#     f_vec = f_vec.reshape(-1, 1)
#     sol = np.linalg.solve(k_matrix, f_vec)
#     return sol


def solve_volterra_leg(k, f, l, n, rng):
    """
    :param k: kernum func
    :param f: right part of eq.
    :param l: lambda
    :param n: num of legendre polynomial
    :param range: (a, b)
    :return: np.array
    """
    nodes, weights = np.polynomial.legendre.leggauss(n)
    grid_np = (rng[1] - rng[0]) * nodes / 2 + (rng[1] + rng[0]) / 2
    k_matrix = np.tril(k(grid_np[:, np.newaxis], grid_np[np.newaxis, :]))
    k_matrix[:] *= weights
    k_matrix *= l
    k_matrix += np.eye(k_matrix.shape[1])
    f_vec = f(grid_np)

    f_vec = f_vec.reshape(-1, 1)
    sol = np.linalg.solve(k_matrix, f_vec)
    return sol, grid_np


def get_k_list(x, y, func, h, mode, a):
    k_list = np.zeros(4)
    if mode != 0:
        k_list[0] = func(x, y, a)
        k_list[1] = func(x + 1 / 3 * h, y + h * k_list[0] * 1 / 3, a)
        k_list[2] = func(x + 2 / 3 * h, y + h * (k_list[1] * 1 - 1 / 3 * k_list[0]), a)
        k_list[3] = func(x + h, y + h * (k_list[2] - k_list[1] + k_list[0]), a)
    else:
        k_list[0] = func(x, y, a)
        k_list[1] = func(x + 1 / 2 * h, y + h * k_list[0] * 1 / 2, a)
        k_list[2] = func(x + 1 / 2 * h, y + h * k_list[1] * 1 / 2, a)
        k_list[3] = func(x + h, y + h * k_list[2], a)
    return k_list


def runge2(funcs, y_ic, x_range, mode, sigma_list, a=0):
    h = x_range[1] - x_range[0]
    y_ic = np.array(y_ic, dtype=float)
    num_eq = len(funcs)
    num_steps = len(x_range)

    y_list = np.zeros((num_eq, num_steps))
    y_list[:, 0] = y_ic

    for idx in range(1, num_steps):
        x_prev = x_range[idx - 1]
        y_prev = y_ic.copy()
        y_new = y_prev.copy()

        for i, func in enumerate(funcs):
            k = get_k_list(x_prev, y_prev, func, h, mode, a)
            summ = np.dot(sigma_list, k)
            y_new[i] += h * summ

        y_ic = y_new
        y_list[:, idx] = y_new

    return y_list