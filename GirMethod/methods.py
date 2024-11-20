import numpy as np


def euler_method(funcs, init_cond, t_grid):
    sol = np.zeros((funcs.shape[0], t_grid.shape[0]))
    sol[:, 0] = init_cond

    for i in range(1, len(t_grid)):
        delta_t = t_grid[i] - t_grid[i - 1]
        var_list = np.concatenate(([t_grid[i - 1]], sol[:, i - 1]))
        for j, func in enumerate(funcs):
            sol[j, i] = sol[j, i - 1] + delta_t * func(var_list)

    return sol


def euler_method2(A_matrix, init_cond, t_grid):
    sol = np.zeros((A_matrix.shape[0], t_grid.shape[0]))
    sol[:, 0] = init_cond

    for i in range(1, len(t_grid)):
        delta_t = t_grid[i] - t_grid[i - 1]
        var_list = np.concatenate(([t_grid[i - 1]], sol[:, i - 1]))
        sol[:, i] = sol[:, i - 1] + delta_t * np.dot(A_matrix, var_list)

    return sol


def modified_euler_method1(funcs, init_cond, t_grid):
    sol = np.zeros((funcs.shape[0], t_grid.shape[0]))
    sol[:, 0] = init_cond

    for i in range(1, len(t_grid)):
        delta_t = t_grid[i] - t_grid[i - 1]
        # prediction
        pred = np.zeros(funcs.shape[0])
        var_list = np.concatenate(([t_grid[i - 1]], sol[:, i - 1]))
        for j, func in enumerate(funcs):
            pred[j] = sol[j, i - 1] + delta_t * func(var_list)
        # correction
        var_list_2 = np.concatenate(([t_grid[i]], pred))
        for j, func in enumerate(funcs):
            sol[j, i] = sol[j, i - 1] + delta_t * (func(var_list_2) + func(var_list)) / 2

    return sol


def get_k_list(x, y, func, h, mode):
    k_list = np.zeros(4)
    if mode != 0:
        k_list[0] = func((x, y))
        k_list[1] = func((x + 1 / 3 * h, y + h * k_list[0] * 1 / 3))
        k_list[2] = func((x + 2 / 3 * h, y + h * (k_list[1] * 1 - 1 / 3 * k_list[0])))
        k_list[3] = func((x + h, y + h * (k_list[2] - k_list[1] + k_list[0])))
    else:
        k_list[0] = func((x, y))
        k_list[1] = func((x + 1 / 2 * h, y + h * k_list[0] * 1 / 2))
        k_list[2] = func((x + 1 / 2 * h, y + h * k_list[1] * 1 / 2))
        k_list[3] = func((x + h, y + h * k_list[2]))
    return k_list


def runge2(funcs, y_ic, x_range, mode, sigma_list):
    y_ic = np.array(y_ic, dtype=float)
    num_eq = len(funcs)
    num_steps = len(x_range)

    y_list = np.zeros((num_eq, num_steps))
    y_list[:, 0] = y_ic

    for idx in range(1, num_steps):
        h = x_range[idx] - x_range[idx - 1]
        x_prev = x_range[idx - 1]
        y_prev = y_ic.copy()
        y_new = y_prev.copy()

        for i, func in enumerate(funcs):
            k = get_k_list(x_prev, y_prev, func, h, mode)
            summ = np.dot(sigma_list, k)
            y_new[i] += h * summ

        y_ic = y_new
        y_list[:, idx] = y_new

    return y_list


def implicit_euler_method(funcs, init_cond, t_grid, max_iter=1e3, delta=1e-6, phi=None):
    x = init_cond.copy()
    x_prev = x
    sol = np.zeros((funcs.shape[0], t_grid.shape[0]))
    sol[:, 0] = init_cond

    for i in range(1, len(t_grid)):
        delta_t = t_grid[i] - t_grid[i - 1]
        t = t_grid[i]

        def phi(x):
            var_list = np.hstack(([t], x))
            f = np.array([func(var_list) for func in funcs])
            return x - x_prev - delta_t * f

        # def phi(x):
        #     var_list = np.concatenate(([t], x))
        #     return x - x_prev - delta_t * np.array([func(var_list) for func in funcs])
        #
        # def jacobian(x):
        #     var_list = np.hstack(([t], x))
        #     jacobian_matrix = np.zeros((len(x), len(x)))
        #     for j in range(len(x)):
        #         x_plus, x_minus = x.copy(), x.copy()
        #         x_plus[j] += delta
        #         x_minus[j] -= delta
        #         var_list_plus = np.hstack(([t + delta_t], x_plus))
        #         var_list_minus = np.hstack(([t - delta_t], x_minus))
        #         jacobian_matrix[:, j] = (
        #             np.array([func(var_list_plus) for func in funcs]) -
        #             np.array([func(var_list_minus) for func in funcs])
        #         ) / (2 * delta)
        #
        #     return np.eye(len(x)) + jacobian_matrix * -delta_t

        def jacobian(x):
            var_list = np.hstack(([t], x))
            f0 = np.array([func(var_list) for func in funcs])
            n = len(x)
            jacobian_matrix = np.zeros((n, n))
            for j in range(n):
                x_perturbed = x.copy()
                x_perturbed[j] += delta
                var_list_perturbed = np.hstack(([t], x_perturbed))
                f_perturbed = np.array([func(var_list_perturbed) for func in funcs])
                jacobian_matrix[:, j] = (f_perturbed - f0) / delta
            return np.eye(n) - delta_t * jacobian_matrix

        for iteration in range(int(max_iter)):
            phi_m = jacobian(x)
            phi_x = phi(x)
            delta_x = np.dot(np.linalg.inv(phi_m), phi_x)
            x = x - delta_x
            err = np.linalg.norm(delta_x, ord=2)
            if err < 1e-6:
                break

        sol[:, i] = x
        x_prev = x

    return sol


def gear_method(funcs, init_cond, t_grid, m, base_method_c, max_iter=1e3, delta=1e-6):
    sol = np.zeros((funcs.shape[0], t_grid.shape[0]))
    b_list = np.array([1])
    a_matrix = np.ones((m + 1, m + 1))
    a_matrix[1:, 0] = 0
    for i, row in enumerate(a_matrix):
        a_matrix[i, 1:] *= np.array([j ** i for j in range(1, m + 1)])

    rb = np.zeros((m + 1, 1))
    rb[1, 0] = -1
    a_list = np.linalg.solve(a_matrix, rb).flatten()

    if type(base_method_c) is str:
        if base_method_c == 'ie':
            first_m_sol = implicit_euler_method(funcs, init_cond, t_grid[:m], max_iter=max_iter, delta=delta)
        elif base_method_c == 'me':
            first_m_sol = modified_euler_method1(funcs, init_cond, t_grid[:m])
        elif base_method_c == 'e':
            first_m_sol = euler_method(funcs, init_cond, t_grid[:m])
        elif base_method_c == 'rk':
            sigma_classic = np.array([1 / 6, 2 / 6, 2 / 6, 1 / 6])
            first_m_sol = runge2(funcs, init_cond, t_grid[:m], 0, sigma_classic)
    else:
        if base_method_c.shape[1] != m:
            raise Exception('Initial sols does not match the m')
        else:
            first_m_sol = base_method_c

    sol[:, :m] += first_m_sol

    x = sol[:, m].flatten()

    for i in range(m, t_grid.shape[0]):
        delta_t = t_grid[i] - t_grid[i - 1]
        t = t_grid[i]

        def phi(x):
            var_list = np.hstack(([t], x))
            f = np.array([func(var_list) for func in funcs])
            res = x.copy()
            for j in range(1, m + 1):
                res += sol[:, i - j] * a_list[j] / a_list[0]

            return res - delta_t * f / a_list[0]

        def jacobian(x):
            var_list = np.hstack(([t], x))
            f0 = np.array([func(var_list) for func in funcs])
            n = len(x)
            jacobian_matrix = np.zeros((n, n))
            for j in range(n):
                x_perturbed = x.copy()
                x_perturbed[j] += delta
                var_list_perturbed = np.hstack(([t], x_perturbed))
                f_perturbed = np.array([func(var_list_perturbed) for func in funcs])
                jacobian_matrix[:, j] = (f_perturbed - f0) / delta
            return np.eye(n) - delta_t * jacobian_matrix

        for iteration in range(int(max_iter)):
            phi_m = jacobian(x)
            phi_x = phi(x)
            delta_x = np.dot(np.linalg.inv(phi_m), phi_x)
            x = x - delta_x
            err = np.linalg.norm(delta_x, ord=2)
            if err < 1e-6:
                break

        sol[:, i] = x

    return sol


def adams_bashforth_moulton_methods(funcs, init_cond, t_grid, base_method_c):
    sol = np.zeros((funcs.shape[0], t_grid.shape[0]))

    sol[:, :4] = base_method_c
    tau = t_grid[1] - t_grid[0]

    for i in range(3, t_grid.shape[0]-1):
        fi = funcs[0](np.hstack(([t_grid[i]], sol[:, i][0])))
        fi1 = funcs[0](np.hstack(([t_grid[i-1]], sol[:, i-1][0])))
        fi2 = funcs[0](np.hstack(([t_grid[i-1]], sol[:, i-1][0])))
        fi3 = funcs[0](np.hstack(([t_grid[i-1]], sol[:, i-1][0])))

        sol[:, i+1] = sol[:, i] + tau/24. * (55*fi - 59*fi1 + 37*fi2 - 9*fi3)
        var_list = np.hstack(([t_grid[i+1]], sol[:, i+1][0]))
        sol[:, i+1] = sol[:, i] + tau/24. * (9*funcs[0](var_list) + 19*fi - 5*fi1 + fi2)

    return sol
