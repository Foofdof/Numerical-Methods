import numpy as np
from functions import (
    solve_fredholm_tr, solve_fredholm_leg,
    solve_volterra_tr, solve_volterra_leg,
    runge2
)
import matplotlib.pyplot as plt
from scipy import interpolate


def p_sol_1(t, l):
    return (36+324*l+13*l**2+48*l*t-18*l**2*t-36*t**2-162*l*t**2+6*l**2*t**2)/(6*(-6-27*l+l**2))


def p_sol_2(t, l):
    return (1+np.exp(t))*(t+np.log(2)-np.log(1+np.exp(t)))/(1+np.exp(t))+np.exp(-t)


def task1():
    t_min = 1
    t_max = 2
    lmbd = 1
    sig_points = [t_min, np.average((t_min, t_max)), t_max]
    funcs = [lambda t, s: t + 2 * s, lambda t: t ** 2 - 1]
    # funcs = [lambda t, s: 1/np.sqrt(t + s**2), lambda t: np.sqrt(t+1)-np.sqrt(t+4)+t]

    n = 10
    grid = np.linspace(t_min, t_max, n)
    sol = solve_fredholm_tr(funcs[0], funcs[1], lmbd, grid)
    sol_g, grid_g = solve_fredholm_leg(funcs[0], funcs[1], lmbd, n, (t_min, t_max))

    n = 3
    grid2 = np.linspace(t_min, t_max, n)
    sol2 = solve_fredholm_tr(funcs[0], funcs[1], lmbd, grid2)
    sol2_g, grid2_g = solve_fredholm_leg(funcs[0], funcs[1], lmbd, n, (t_min, t_max))

    grid_p = np.linspace(t_min, t_max, 100)

    tck_1 = interpolate.splrep(grid_g.flatten(), sol_g.flatten(), k=3)
    tck_1_tr = interpolate.splrep(grid, sol, k=3)
    tck_2 = interpolate.splrep(grid2_g.flatten(), sol2_g.flatten(), k=2)
    # tck_2_tr = interpolate.splrep(grid2_g.flatten(), sol2_g.flatten(), k=2)

    fig, ax = plt.subplots(1)
    ax.scatter(grid, sol, label='$x(t), n=10$', color='blue', marker='o')
    ax.scatter(grid2, sol2, label='$x(t), n=3$', color='blue', marker='^')

    ax.scatter(grid_g, sol_g, label='$x(t), n=10$', color='black', marker='o')
    ax.scatter(grid2_g, sol2_g, label='$x(t), n=3$', color='gray', marker='^')
    ax.plot(grid_p, interpolate.splev(grid_p, tck_1, der=0), label='$x(t)_{int}$ $n=10$', color='black')
    ax.plot(grid_p, interpolate.splev(grid_p, tck_2, der=0), label='$x(t)_{int}$ $n=3$', color='gray')

    ax.plot(grid_p, p_sol_1(grid_p, lmbd), label='$x_p(t), $', color='red')

    ax.set_xlabel('t')
    ax.set_ylabel('x(t)')
    ax.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    plt.title('Fredholm int. eq. solution', fontsize=16, fontweight='bold')
    plt.show()

    fig, ax = plt.subplots(1)
    ax.plot(grid_p, np.abs(p_sol_1(grid_p, lmbd) - interpolate.splev(grid_p, tck_1, der=0)),
            label='$x(t)_{int}$ $n=10$ leg', color='black')
    ax.plot(grid_p, np.abs(p_sol_1(grid_p, lmbd) - interpolate.splev(grid_p, tck_1_tr, der=0)),
            label='$x(t)_{int}$ $n=10$ tr', color='blue')
    ax.plot(grid_p, np.abs(p_sol_1(grid_p, lmbd) - interpolate.splev(grid_p, tck_2, der=0)),
            label='$x(t)_{int}$ $n=10$ tr', color='gray')

    ax.set_xlabel('t')
    ax.set_ylabel('x(t)')
    ax.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    plt.title('Abs Error', fontsize=16, fontweight='bold')
    plt.show()

    print("n=10: ", interpolate.splev(sig_points, tck_1, der=0))
    print("n=3: ", interpolate.splev(sig_points, tck_2, der=0))


def task2():
    t_min = 0
    t_max = 4
    lmbd = -1
    tau = 0.2
    sigma_classic = np.array([1 / 6, 2 / 6, 2 / 6, 1 / 6])
    # sig_points = [t_min, np.average((t_min, t_max)), t_max]
    funcs = [lambda t, s: np.exp(s)/(np.exp(t)+1), lambda t: np.exp(-t)]
    # funcs2 = [lambda t, s: np.exp(s) / (np.exp(t) + 1) * np.sqrt(1-((s-(t_max+t_min)/2)*2/(t_max-t_min))**2), lambda t: np.exp(-t)]
    n = np.int64(np.round(t_max - t_min)/tau)+1
    grid = np.arange(t_min, t_max+tau, tau)
    sol = solve_volterra_tr(funcs[0], funcs[1], lmbd, grid)
    # sol_g, grid_g = solve_volterra_leg(funcs[0], funcs[1], lmbd, n, (t_min, t_max))

    grid_p = np.linspace(t_min, t_max, 100)

    funcs = [lambda t, u, a: np.exp(t) * u[0] / (np.exp(t) + 1) + 1]
    u_list = runge2(funcs, [0], grid, 0, sigma_classic)
    sol_ode = u_list[0]/(np.exp(grid) + 1) + np.exp(-grid)
    # print(sol_ode[0], sol_ode[-1])

    tck_1 = interpolate.splrep(grid, sol, k=3)
    tck_2 = interpolate.splrep(grid, sol_ode, k=3)

    fig, ax = plt.subplots(1)
    ax.scatter(grid, sol, label=f'$x(t), n={n}$'+', трапеции', color='blue')
    ax.scatter(grid, sol_ode, label=f'$x(t), n={n}$'+', ОДУ', color='red')
    ax.plot(grid_p, interpolate.splev(grid_p, tck_1, der=0), label=None, color='blue')
    ax.plot(grid_p, interpolate.splev(grid_p, tck_2, der=0), label=None, color='red')
    ax.plot(grid_p, p_sol_2(grid_p, lmbd), label=f'$x(t)' + ', точное решение', color='green')

    ax.set_xlabel('t')
    ax.set_ylabel('x(t)')
    ax.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    plt.title('Volterra int. eq. solution', fontsize=16, fontweight='bold')
    plt.show()

    fig, ax = plt.subplots(1)
    ax.plot(grid_p, np.abs(p_sol_2(grid_p, lmbd) - interpolate.splev(grid_p, tck_1, der=0)), label='трапеции', color='blue')
    ax.plot(grid_p, np.abs(p_sol_2(grid_p, lmbd) - interpolate.splev(grid_p, tck_2, der=0)), label='ОДУ', color='red')

    ax.set_xlabel('t')
    ax.set_ylabel('x(t)')
    ax.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    plt.title('Abs Error', fontsize=16, fontweight='bold')
    plt.show()


if __name__ == "__main__":
    task1()
    # task2()
