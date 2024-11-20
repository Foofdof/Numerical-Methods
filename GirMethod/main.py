import numpy as np
from methods import (
    euler_method, euler_method2,
    modified_euler_method1,
    implicit_euler_method,
    gear_method, runge2,
    adams_bashforth_moulton_methods,
)
import matplotlib.pyplot as plt

colors = {
    0: 'red',
    1: 'blue',
    2: 'green',
    3: 'yellow',
}
A_matrix = np.array([
    [0, -1, 100],
    [0, 0, -100]
])
# var_list = [t, x1, x2, x3 ...]
# funcs = [dx1/dt = F1(vl), dx2/dt = F2(vl) ...]
funcs = np.array([
    lambda var_list: -var_list[1] + 100 * var_list[2],
    lambda var_list: -100 * var_list[2],
])
initial_conditions = np.array([1, 1], dtype=np.float64)
sigma_classic = np.array([1 / 6, 2 / 6, 2 / 6, 1 / 6])


def task_1(a, b, tau):
    t_grid = np.arange(a, b + tau / 2, tau)

    sol1 = euler_method(funcs, initial_conditions, t_grid)
    sol2 = euler_method2(A_matrix, initial_conditions, t_grid)
    pr_sol = np.array([
        1 / 99. * np.exp(-100 * t_grid) * (-100 + 199 * np.exp(99 * t_grid)),
        np.exp(-100 * t_grid)
    ])
    fig, ax = plt.subplots(4)
    for j, sol in enumerate([sol1, sol2, np.abs(sol1 - pr_sol), np.abs(sol2 - pr_sol)]):
        for i in range(sol.shape[0]):
            ax[j].plot(t_grid, sol[i], c=colors[i], label=f'$x_{i + 1}(t)$')

    for a in ax:
        a.legend()
        a.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    fig.suptitle(f'Euler, tau {tau}', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    plt.show()


def task_2(a, b, tau):
    t_grid = np.arange(a, b + tau / 2, tau)

    sol1 = modified_euler_method1(funcs, initial_conditions, t_grid)
    pr_sol = np.array([
        1 / 99. * np.exp(-100 * t_grid) * (-100 + 199 * np.exp(99 * t_grid)),
        np.exp(-100 * t_grid)
    ])
    fig, ax = plt.subplots(2)
    for j, sol in enumerate([sol1, np.abs(sol1 - pr_sol)]):
        for i in range(sol.shape[0]):
            ax[j].plot(t_grid, sol[i], c=colors[i], label=f'$x_{i + 1}(t)$')

    for a in ax:
        a.legend()
        a.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    fig.suptitle(f'Modified Euler, tau {tau}', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    plt.show()


def task_3(a, b, tau):
    t_grid = np.arange(a, b + tau / 2, tau)

    sol1 = implicit_euler_method(funcs, initial_conditions, t_grid)
    pr_sol = np.array([
        1 / 99. * np.exp(-100 * t_grid) * (-100 + 199 * np.exp(99 * t_grid)),
        np.exp(-100 * t_grid)
    ])
    fig, ax = plt.subplots(2)
    for j, sol in enumerate([sol1, np.abs(sol1 - pr_sol)]):
        for i in range(sol.shape[0]):
            ax[j].plot(t_grid, sol[i], c=colors[i], label=f'$x_{i + 1}(t)$')

    for a in ax:
        a.legend()
        a.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    fig.suptitle(f'Implicit Euler, tau {tau}', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    plt.show()


def task_4(a, b, tau, m, base_method_c='ie'):
    t_grid = np.arange(a, b + tau / 2, tau)

    sol1 = gear_method(
        funcs=funcs,
        init_cond=initial_conditions,
        t_grid=t_grid,
        base_method_c=base_method_c,
        m=m,
    )

    pr_sol = np.array([
        1 / 99. * np.exp(-100 * t_grid) * (-100 + 199 * np.exp(99 * t_grid)),
        np.exp(-100 * t_grid)
    ])
    fig, ax = plt.subplots(2)
    for j, sol in enumerate([sol1, np.abs(sol1 - pr_sol)]):
        for i in range(sol.shape[0]):
            ax[j].plot(t_grid, sol[i], c=colors[i], label=f'$x_{i + 1}(t)$')

    for a in ax:
        a.legend()
        a.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    fig.suptitle(f'Implicit Gear, tau: {tau}, m: {m}', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    return sol1


def part2(a, b, tau, m, base_method_c='ie'):
    funcs = np.array([


        lambda var_list: 2*np.sin(var_list[0]) + var_list[1],
    ])
    initial_conditions = np.array([1], dtype=np.float64)

    t_grid = np.arange(a, b + tau / 2, tau)

    sol1 = runge2(
        funcs=funcs,
        y_ic=initial_conditions,
        x_range=t_grid,
        mode=0,
        sigma_list=sigma_classic
    )

    sol_a = adams_bashforth_moulton_methods(funcs, initial_conditions, t_grid, sol1[0, :4])[0]

    sol_g = gear_method(
        funcs=funcs,
        init_cond=initial_conditions,
        t_grid=t_grid,
        base_method_c='rk',
        m=m,
    )[0]

    pr_sol = np.array(
        np.exp(t_grid)*(1+1)-np.sin(t_grid)-np.cos(t_grid)
    )

    fig, ax = plt.subplots(2)
    ax[0].plot(t_grid, sol_a, c='red', label=f'$x_a(t)$')
    ax[0].plot(t_grid, sol_g, c='black', label=f'$x_g(t)$')
    ax[0].plot(t_grid, pr_sol, c='blue', label=f'$x_p(t)$')

    ax[1].plot(t_grid, np.abs(sol_a-pr_sol), c='red', label=f'$x_a(t)$')
    ax[1].plot(t_grid, np.abs(sol_g-pr_sol), c='black', label=f'$x_g(t)$')

    for a in ax:
        a.legend()
        a.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    fig.suptitle('Adams-Bashforth-Moulton', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    return sol1


if __name__ == '__main__':
    # part 1

    # for tau in (0.1, 0.01, 0.001):
    #     task_1(0, 1, tau)
    #     task_2(0, 1, tau)
    #     task_3(0, 1, tau)

    # for m in (1, 2, 4):
    #     task_4(0, 1, 0.01, m, base_method_c='ie')
    #
    # gear_1 = task_4(0, 1, 0.01, 1)
    # gear_2 = task_4(0, 1, 0.01, 2, base_method_c=gear_1[:, :2])
    # gear_4 = task_4(0, 1, 0.01, 4, base_method_c=gear_2[:, :4])

    # part 2
    part2(0, 1, 0.01, 4)
