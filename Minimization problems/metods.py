from numpy import *
from functions import derivative, random_point, make_dec, neighbour, grad


def golden_cut(eps, func, start, end):
    """returns vector of x_min"""
    ratio = (1 + 5 ** (1 / 2)) / 2
    x2 = array(end) - (array(end) - array(start)) / ratio
    x1 = array(start) + (array(end) - array(start)) / ratio
    while True:
        if func(x1) <= func(x2):
            start = x1
            if linalg.norm(array(end) - array(start)) < eps:
                break
            else:
                start = x2
                end = array(end) - (array(end) - array(start)) / ratio
        else:
            if linalg.norm(array(end) - array(start)) < eps:
                break
            else:
                end = x1
                x1 = array(start) + (array(end) - array(start)) / ratio

    return (x1 + x2) / 2


def newton(eps, func, start, end):
    """returns vector of x_min"""
    x1 = (array(start) + array(end)) / 2
    x2 = x1 - derivative(func, x1, 1) / derivative(func, x1, 2)
    while linalg.norm(array(x2) - array(x1)) > eps:
        x1 = x2
        x2 = x1 - derivative(func, x1, 1) / derivative(func, x1, 2)

    return x2


def annealing(k_max, t0, func, start, end):
    x_old = random_point(start, end)
    x_new = x_old
    k = 1
    while k <= k_max:
        t = t0 / k
        x_new = neighbour(x_old, t)
        e_old = func(x_old)
        e_new = func(x_new)

        if make_dec(e_old, e_new, t) >= random.rand():
            x_old = x_new
        k += 1
    return x_new


def gradient_descent(eps, func, start, end, t):
    d = len(start)
    x0 = (array(start) + array(end)) / 2
    x1 = zeros(d)
    for i in range(d):
        to_min = lambda alpha: func((x0 - alpha * x0).tolist())
        t = golden_cut(0.001, to_min, [t / 2], [t * 2])[0]
        x1[i] = x0[i] - t / (i + 1) * grad(func, x0.tolist(), i)

    while max(abs(x1 - x0)) > eps:
        x0 = x1
        for i in range(d):
            to_min = lambda alpha: func((x0 - alpha * x0).tolist())
            t = golden_cut(0.001, to_min, [t / 2], [t * 2])[0]
            x1[i] = x0[i] - t / (i + 1) * grad(func, x0.tolist(), i)
            t = t / (i + 1)

    return x1.tolist()


def matrix_solve(eps, a_mtr, b_mtr):
    # x = x0 = zeros(len(b_mtr))
    x = x0 = b_mtr
    r = r0 = b_mtr - dot(a_mtr, x0)
    p0 = r0
    norm_r = linalg.norm(r)
    while norm_r > eps:
        alpha = dot(r0.transpose(), r0) / dot(p0.transpose(), dot(a_mtr, p0))
        x = x0 + alpha * p0
        r = r0 - alpha * dot(a_mtr, p0)
        beta = dot(r.transpose(), r) / dot(r0.transpose(), r0)

        norm_r = linalg.norm(r)
        p0 = r + beta * p0
        r0 = r
        x0 = x

    return x
