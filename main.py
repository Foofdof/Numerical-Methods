from functions import *
import matplotlib.pyplot as plt


def main():
    print("Task 1:")
    # dichotomy investigation
    xd_list = dichotomy(1, 3, 0.00001, func)
    yd_list = list(map(func, xd_list))

    # relaxation investigation
    xr_list = relax(1, 3, 0.00001, phi)
    yr_list = list(map(func, xr_list))

    # Newton investigation
    _, xn_list = newton(1, 3, 0.00001, func)
    yn_list = list(map(func, xn_list))

    # Plot assembling
    x_list2 = numpy.linspace(1, 2, 50)
    y_list2 = list(map(func, x_list2))
    fig, ax = plt.subplots(3)
    ax[0].scatter(xr_list, yr_list, c="red", label="Relaxation")
    ax[1].scatter(xd_list, yd_list, c="black", label="Dichotomy")
    ax[2].scatter(xn_list, yn_list, c="darkblue", label="Newton")
    for a in ax:
        a.plot(x_list2, y_list2, label="f(x)")
        a.legend()

    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    print("Task 2:")
    x_list = numpy.arange(0.01, 13, 0.05)
    for i in range(len(x_list) - 1):
        if lager(x_list[i]) * lager(x_list[i + 1]) < 0:
            a = x_list[i] - 0.1
            b = x_list[i + 1] + 0.1
            print("Roots: ")
            dichotomy(a, b, 0.00001, lager)
            # relax(a, b, 0.00001, phi_lager)
            newton(a, b, 0.00001, lager)

    print("Task 3:")
    root, *_ = newton(2.8, 3.5, 0.001, multy_root)

    x_list = numpy.arange(2.8, 3.5, 0.1)
    y_list = list(map(multy_root, x_list))
    n = len(x_list)
    c_matrix = numpy.eye(n, n)
    for i in range(len(c_matrix)):
        for j in range(len(c_matrix)):
            c_matrix[i][j] = (x_list[i]) ** j
    p = numpy.poly1d(numpy.flip(numpy.linalg.solve(c_matrix, y_list)))
    n = 0
    print()
    while p(root) < 0.005:
        p = numpy.polyder(p)
        n = n + 1
    print("Another method(polynomial): ", n)


if __name__ == "__main__":
    main()
