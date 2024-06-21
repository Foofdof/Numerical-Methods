import numpy as np
from matplotlib.animation import FuncAnimation

from methods import explicit_scheme, implicit_scheme
import matplotlib.pyplot as plt
from functions import *





def main():
    h = 0.01
    tau = 0.005
    xl = np.linspace(0, 1, int((1 - 0) / h) + 1, dtype=np.float64)
    tl = np.linspace(0, 1, int((1 - 0) / tau) + 1, dtype=np.float64)
    region_list = [xl, tl]
    func = [lambda x: 2 * x * (x - 1), lambda x: 1 + np.exp(x), lambda t: np.sin(np.pi * t), lambda t: 2 * t ** 3]
    mtr = make_matrix(func, region_list)
    f = np.zeros((len(mtr), len(mtr[0])))
    for i in range(0, len(mtr)):
        for j in range(0, len(mtr[0])):
            f[i][j] = (0.55 * (tl[i]) ** 2 + 5) / (1 + (0.05 * xl[j]) ** 2)

    sol = explicit_scheme(np.copy(mtr), np.sqrt(1), h, tau, f)
    sol2 = implicit_scheme(np.copy(mtr), np.sqrt(1), h, tau, f, region_list, func, 0.35)
    xgrid, tgrid = np.meshgrid(xl, tl)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(tgrid, xgrid, sol2 - sol, cmap='cividis')
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_zlabel('U')
    plt.show()
    i: int
    for i in range(0, len(mtr)):
        for j in range(0, len(mtr[0])):
            print('{', str(tl[i]), ',', str(xl[j]), ',', str(sol2[i, j]), '},')

    fig, ax = plt.subplots()
    ax.plot(region_list[0], sol2[0], label="u(t)")
    ax.legend()

    def update(s):
        plt.clf()
        # line.set_data(region_list[0], sol2[s])
        return ax.plot(region_list[0], sol2[s], 'Black', label="u(t)")

    anim = FuncAnimation(fig, func=update, frames=len(sol2), interval=50, blit=True)
    plt.show()


if __name__ == "__main__":
    main()
