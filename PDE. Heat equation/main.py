from methods import *
from functions import make_matrix
import matplotlib.pyplot as plt


def main():
    print("1")
    h = 0.05
    tau = 1e-4
    xl = np.linspace(0, 1, int((1 - 0) / h))
    tl = np.linspace(0, 0.1, int((0.1 - 0) / tau))
    region_list = [xl, tl]
    func = [lambda x: 1+np.sin(np.pi*x), lambda t: np.cos(t), lambda t: np.cos(t)]
    mtr = make_matrix(func, region_list)
    f = np.zeros((len(mtr), len(mtr[0])))
    for i in range(0, len(mtr)):
        for j in range(0, len(mtr[0])):
            f[i][j] = - np.sin(tl[i])
    sol = explicit_scheme(mtr, np.sqrt(1/2), h, tau, f)
    sol2 = implicit_scheme(func, region_list, np.sqrt(1 / 2), h, tau)

    sol3 = np.zeros((len(mtr), len(mtr[0])))
    for i in range(0, len(mtr)):
        for j in range(0, len(mtr[0])):
            sol3[i][j] = np.exp(-(np.pi)**2*tl[i]/2)*np.sin(np.pi*xl[j])

    xgrid, tgrid = np.meshgrid(xl, tl)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(tgrid, xgrid, sol, cmap='cividis')
    ax.plot_surface(tgrid, xgrid, sol-sol3, cmap='cividis')
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_zlabel('U')
    plt.show()


if __name__ == '__main__':
    main()
