import numpy as np


def first(x, y_args, a=0):
    return y_args[1]


def second(x, y_args, a=0):
    return -1 * y_args[0]


def get_k_list(x, y, func, h, mode, a):
    k_list = np.zeros(4)
    if mode != 0:
        k_list[0] = func(x, y, a)
        k_list[1] = func(x + 1 / 3 * h, y + h * k_list[0] * 1 / 3, a)
        k_list[2] = func(x + 2 / 3 * h, y + h * (k_list[1] * 1 - 1 / 3 * k_list[0]), a)
        k_list[3] = func(x + h, y + h * (k_list[2] - k_list[1] + k_list[0]), a)
    else:
        k_list[0] = func(x, y, a)
        k_list[1] = func(x + 1 / 2 * h, y + h * k_list[0] * 1 / 2, a)
        k_list[2] = func(x + 1 / 2 * h, y + h * k_list[1] * 1 / 2, a)
        k_list[3] = func(x + h, y + h * k_list[2], a)
    return k_list


def first_t(x, y_args, a=0):
    return y_args[1]


def second_t(x, y_args, a=0.5):
    return -(a * (y_args[0] * y_args[0] - 1) * y_args[1] + y_args[0])
