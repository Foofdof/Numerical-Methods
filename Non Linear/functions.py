import numpy
from math import *


def multy_root(x):
    return (x - 3) ** 2


def func(x):
    return 2 * log(x) + sin(log(x)) - cos(log(x))


def lager(x):
    return 1 / 120 * (-x ** 5 + 25 * x ** 4 - 200 * x ** 3 + 600 * x ** 2 - 600 * x + 120)


def phi_lager(x):
    return (-x ** 5 + 25 * x ** 4 - 200 * x ** 3 + 600 * x ** 2 + 120) / 600


def phi(x):
    return numpy.exp((-sin(log(x)) + cos(log(x))) / 2)


def dichotomy(a, b, eps, f):
    n = 0
    x_list = []
    while (b - a) > eps:
        c = (a + b) / 2
        x_list.append(c)
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
        n = n + 1

    print("Root: ", (a + b) / 2, " n: ", n, "k(f): ", 2 * n)
    return x_list


def relax(a, b, eps, f):
    n = 1
    xp = (a + b) / 2
    x = f(xp)
    cur_eps = abs(x - xp)
    x_list = [x]
    while cur_eps > eps:
        xp = x
        x = f(x)
        x_list.append(x)
        cur_eps = abs(x - xp)
        n = n + 1

    print("Root: ", x, " n: ", n, "k(f): ", n)
    return x_list


def newton(a, b, eps, f):
    k = 1
    h = (a + b) / 100
    x_list = [0, 0, 0]
    x_list[0] = (a + b) / 2
    x_list[1] = x_list[0] - f(x_list[0]) * h / (f(x_list[0] + h / 2) - f(x_list[0] - h / 2))
    x_list[2] = x_list[1] - f(x_list[1]) * h / (f(x_list[1] + h / 2) - f(x_list[1] - h / 2))
    cur_eps = abs(x_list[2] - x_list[1])
    while cur_eps > eps:
        n = len(x_list) - 1
        x_list.append(x_list[n] - f(x_list[n]) * h / (f(x_list[n] + h / 2) - f(x_list[n] - h / 2)))
        cur_eps = abs(x_list[n + 1] - x_list[n])
        k = k + 1

    x = x_list[len(x_list) - 1]
    q = (x - x_list[len(x_list) - 2]) / (x_list[len(x_list) - 2] - x_list[len(x_list) - 3])
    p = 1 / (1 - q)
    print("Root: ", x, " n: ", k, "k(f): ", 3 * k, " p:", p)
    return x, x_list
