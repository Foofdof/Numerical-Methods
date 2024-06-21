from methods import debc
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp


def main():
    g_list = [1, 0, 1, 1, 0, 0]
    a = 1
    b = 10
    nu = 2
    funcs = [lambda x: 1 / x, lambda x: (x ** 2 - nu**2) / x ** 2, lambda x: 0]
    x_list, y_list = debc(g_list, [a, b], funcs, 0.001)
    y_list2 = sp.special.jn(nu, x_list)
    fig, ax = plt.subplots(2)
    ax[0].plot(x_list, y_list, label='$u(x)$')
    ax[0].plot(x_list, y_list2, label=r'$J_%s (x) $'%nu)
    r = list(zip(x_list, y_list))
    print((','.join(str(i) for i in r)).replace('(', '{').replace(')', '}'))

    mu1 = 1
    mu2 = 1
    beta1 = 0
    beta2 = 0
    g_list = [0, -a**2, mu1-beta1, 0, b**2, mu2-beta2]
    funcs = [lambda x: 1 / x, lambda x: (x ** 2 - nu ** 2) / x ** 2, lambda x: 0]
    x_list, y_list = debc(g_list, [1, 10], funcs, 0.001)
    ax[1].plot(x_list, y_list, label='$u(x)$')
    for a in ax:
        a.legend()
    plt.show()
    print("Plot printed")

    r = list(zip(x_list, y_list))
    print((','.join(str(i) for i in r)).replace('(','{').replace(')','}'))


if __name__ == '__main__':
    main()
