import numpy as np

from functions import make_matrix
from methods import *
import matplotlib.pyplot as plt


def main():
    print("Task 1")
    h = np.pi/2/100
    l = np.pi/2/100
    # xl = np.linspace(0, 1 / 2, int((1 / 2 - 0) / h))
    # yl = np.linspace(np.pi / 2, np.pi, int((np.pi - np.pi / 2) / l))
    # region_list = [xl, yl]
    # func = [lambda y: np.sin(y), lambda y: 1.64872 * np.sin(y), lambda x: np.exp(x), lambda x: 0]
    # mtr = make_matrix(func, region_list)

    xl = np.linspace(0, np.pi/2, int((np.pi/2 - 0) / h), dtype=np.float64)
    yl = np.linspace(0, 1, int((1 - 0) / l), dtype=np.float64)
    region_list = [xl, yl]
    func = [lambda y: np.sinh(y), lambda y: 0, lambda x: 0, lambda x: np.sinh(1)*np.cos(x)]
    mtr = make_matrix(func, region_list)
    print(mtr)
    sol, k = pde_ziedel_solve(np.copy(mtr), h, l, 0.00001)
    print(sol)
    print(k)

    sol2, k = pde_relax_solve(np.copy(mtr), h, l, 0.00001, 1.604)
    print(k)

    ygrid, xgrid = np.meshgrid(yl, xl)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(xgrid, ygrid, sol, cmap='Greens')
    ax.plot_surface(xgrid, ygrid, sol2-sol, cmap='Blues')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('U')
    plt.show()

    n = 25
    eps = np.zeros(n+1)
    k_l = np.zeros(n+1)
    for i in range(1, n+1):
        eps[i-1] = 2**(-i)
        _, k = pde_ziedel_solve(np.copy(mtr), h, l, eps[i-1])
        k_l[i-1] = k

    fig, ax = plt.subplots(1)
    ax.scatter(eps, k_l, label='$k(log[eps])$')
    ax.set_xscale('log')
    ax.set_xlabel('log[eps]')
    ax.set_ylabel('k')
    ax.legend()
    plt.show()

    w_l = np.arange(1.897, 1.913, 0.001, dtype=np.float64)
    k_l = np.zeros(len(w_l), dtype=np.float64)

    h = np.pi / 2 / 100
    l = np.pi / 2 / 100
    xl = np.linspace(0, np.pi / 2, int((np.pi / 2 - 0) / h), dtype=np.float64)
    yl = np.linspace(0, 1, int((1 - 0) / l), dtype=np.float64)
    region_list = [xl, yl]
    mtr = make_matrix(func, region_list)

    for i in range(0, len(w_l)):
        _, k = pde_relax_solve(np.copy(mtr), h, l, float(0.001), w_l[i])
        k_l[i] = k
    print(np.where(k_l == k_l.min())[0][0])
    fig, ax = plt.subplots(1)
    ax.scatter(w_l, k_l, label='$k(w)$')
    ax.set_xlabel('w')
    ax.set_ylabel('k')
    ax.legend()
    plt.show()


if __name__ == '__main__':
    main()
