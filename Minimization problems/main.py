from metods import *
from functions import *
import matplotlib.pyplot as plt
import time


def main():
    func_list = [sphere, schwefel_first, schwefel_second, schwefel_third, rosenbrock, steps, rastrigin]
    func_list2 = [rastrigin, steps]
    print("rosenbrock, d = 2")
    print(golden_cut(0.001, rosenbrock, [-100, -100], [100, 100]))

    for func in func_list:
        print(func, ": ")
        try:
            n = 1000
            t0 = time.time()
            print(golden_cut(0.001, func, [-100], [100]))
            for i in range(n):
                golden_cut(0.001, func, [-100], [100])
            print((time.time() - t0) / n)

        except ZeroDivisionError:
            print("ZeroDivisionError")
        try:
            n = 1000
            t0 = time.time()
            print(newton(0.001, func, [-100], [100]))
            for i in range(n):
                newton(0.001, func, [-100], [100])
            print((time.time() - t0) / n)
        except ZeroDivisionError:
            print("ZeroDivisionError")

        print("gradient: ", gradient_descent(0.001, func, [-100, -100], [100, 100], 0.3))

    print("Annealing")
    for func in func_list2:
        print(func, " 2 :")
        try:
            n = 2
            t0 = time.time()
            print(annealing(5000, 100, func, [-100, -100], [100, 100]))
            for i in range(n):
                annealing(5000, 100, func, [-100, -100], [100, 100])
            print((time.time() - t0) / n)
        except ZeroDivisionError:
            print("ZeroDivisionError")


if __name__ == '__main__':
    main()
    print("Matrix solution")
    a_mtr, b_mtr = make_matrix(5)
    print(dot(a_mtr, b_mtr))
    print(matrix_solve(0.001, a_mtr, b_mtr))