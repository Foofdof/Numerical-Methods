import numpy.random
from numpy import *


def test_f(x_vector):
    result = 1
    for x in x_vector:
        result *= x
    return result


def sphere(x_vector):
    """x_d -- vector of cords xi in d-dimension space"""
    result = 0
    for x in x_vector:
        result += x ** 2
    return result


def schwefel_first(x_vector):
    """x_d -- vector of cords xi in d-dimension space"""
    result = 0
    for x in x_vector:
        result += abs(x)
    result += abs(prod(x_vector))
    return result


def schwefel_second(x_vector):
    """x_d -- vector of cords xi in d-dimension space"""
    result = 0
    for i in range(len(x_vector)):
        temp_res = 0
        for j in range(i):
            temp_res += x_vector[j]
        result += temp_res ** 2

    return result


def schwefel_third(x_vector):
    """x_d -- vector of cords xi in d-dimension space"""
    return abs(max(x_vector))


def rosenbrock(x_vector):
    """x_d -- vector of cords xi in d-dimension space"""
    result = 0
    for i in range(len(x_vector) - 1):
        result += 100 * (x_vector[i + 1] - x_vector[i] ** 2) ** 2 + (x_vector[i] - 1) ** 2
    return result


def steps(x_vector):
    """x_d -- vector of cords xi in d-dimension space"""
    result = 0
    for x in x_vector:
        result += floor(x + 0.5) ** 2
    return result


def rastrigin(x_vector):
    """x_d -- vector of cords xi in d-dimension space"""
    result = 0
    for x in x_vector:
        result += x ** 2 - 10 * cos(2 * pi * x) + 10
    return result


def derivative(func, x_vector, k):
    h = 0.001
    if k == 1:
        return (func(array(x_vector) + h) - func(array(x_vector) - h)) / (2 * h)
    if k == 2:
        return (func(array(x_vector) + h) - 2 * func(array(x_vector)) + func(array(x_vector) - h)) / h**2


def gradient_d(func, x_vector, k):
    d = len(x_vector)  # dimension of space
    start = array(x_vector) - 1

    x_range_list = []
    y_range_list = []
    polynomial_list = []
    der_vector = []
    for i in range(d):
        x_range_list.clear()
        temp_list = start.tolist()
        x_list = []
        for j in range(10):
            temp_list[i] = temp_list[i] + 2 / 10
            x_list.append(temp_list[i])
            x_range_list.append(list(temp_list))

        y_range_list.append(list(map(func, x_range_list)))
        y_list = list(y_range_list[i])
        n = len(x_list)
        c_matrix = eye(n, n)
        for c in range(len(c_matrix)):
            for j in range(len(c_matrix)):
                c_matrix[c][j] = (x_list[c]) ** j
        polynomial_list.append(poly1d(flip(linalg.solve(c_matrix, y_list))))

    for i in range(len(x_vector)):
        der_vector.append(polyder(polynomial_list[i], k)(x_vector[i]))

    print(der_vector)


def grad(func, x_vector, i):
    x1 = list(x_vector)
    x2 = x1.copy()
    x1[i] = x1[i] - 0.0000001
    x2[i] = x2[i] + 0.0000001
    return (func(x2) - func(x1)) / 0.0000002


def random_point(start, end):
    d = len(start)
    t_list = random.rand(1, d).tolist()[0]
    for i in range(d):
        t_list[i] = (end[i] - start[i])/2.0 * t_list[i] + (end[i] + start[i])/2.0
    return t_list


def make_dec(e_old, e_new, temperature):
    return exp(-1. * (e_new - e_old) / temperature)


def neighbour(x_old, temperature):
    x_new = []
    for i in x_old:
        x_new.append(i + temperature*random.standard_cauchy())
    return x_new


def make_matrix(n):
    mtr = zeros((n, n))
    b_mtr = zeros(n)
    for i in range(len(mtr)):
        mtr[i, i] = 2
        b_mtr[i] = i**2
        for j in range(len(mtr[i])):
            if i != j:
                mtr[i, j] = 1 / abs(i - j)

    return mtr, b_mtr
