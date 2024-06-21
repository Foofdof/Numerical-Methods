import numpy as np
from functions import get_k_list


def euler(funcs, y_ic, x_range, a=0):
    """
        eps -- error
        funcs -- list of functions in ODE system
        y_ic -- initial conditions for each y (yi(x0))
        x_ic -- x0
    """
    h = x_range[1] - x_range[0]
    y_list = []
    for i in range(len(funcs)):
        y_list.append([])

    y_new = y_ic
    for x in x_range:
        for i in range(len(funcs)):
            y_new[i] = y_new[i] + h * funcs[i](x, y_new, a)
        for i in range(len(y_list)):
            y_list[i].append(y_new[i])
    return y_list


def runge(funcs, y_ic, x_range, mode, sigma_list, a=0):
    """
        eps -- error
        funcs -- list of functions in ODE system
        y_ic -- initial conditions for each y (yi(x0))
        x_ic -- x0
    """
    h = x_range[1] - x_range[0]
    y_list = []
    for i in range(len(funcs)):
        y_list.append([])

    y_new = y_ic
    for x in x_range:
        for i in range(len(funcs)):
            summ = 0
            k_list = get_k_list(x, y_ic, funcs[i], h, mode, a)
            for j in range(len(sigma_list)):
                summ += sigma_list[j] * k_list[j]
            y_new[i] = y_ic[i] + h * summ

        y_ic = y_new
        for i in range(len(y_list)):
            y_list[i].append(y_new[i])
    return y_list


def richardson(eps, funcs, y_ic, x_range, mode, sigma_list, a=0):
    x = x_range[0]
    x_list = []

    h = 0.01
    y_list = []
    for i in range(len(funcs)):
        y_list.append([])

    y_new = y_ic
    while x < x_range[1]:
        for i in range(len(funcs)):
            summ = 0
            while True:
                summ = 0
                k_list = get_k_list(x, y_ic, funcs[i], h, mode, a)
                for j in range(len(sigma_list)):
                    summ += sigma_list[j] * k_list[j]
                if h * summ < eps:
                    break
                else:
                    h = h / 2
            y_new[i] = y_ic[i] + h * summ

        y_ic = y_new
        for i in range(len(y_list)):
            y_list[i].append(y_new[i])
        x_list.append(x)
        x = x+h
    return x_list, y_list
