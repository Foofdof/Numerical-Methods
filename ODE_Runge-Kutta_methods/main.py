from methods import *
from functions import *
import numpy as np
import matplotlib.pyplot as plt


def task_1():
    print("Task 1", "*" * 100)

    print("Euler", "_" * 100)
    y_list = euler([first, second], [1, 0], np.arange(0, 5, 0.05))
    fig, ax = plt.subplots(2)
    ax[0].scatter(np.arange(0, 5, 0.05), y_list[0], c="red", label="x(t)")
    ax[0].plot(np.arange(0, 5, 0.05), y_list[0], 'g-', label="x(t)")
    ax[1].scatter(np.arange(0, 5, 0.05), y_list[1], c="black", label="x'(t)")
    for a in ax:
        a.legend()
    plt.show()
    print("Plot printed")

    print("Runge", "_" * 100)
    sigma_classic = np.array([1 / 6, 2 / 6, 2 / 6, 1 / 6])
    sigma_38 = np.array([1 / 8, 3 / 8, 3 / 8, 1 / 8])
    y_list = runge([first, second], [1, 0], np.arange(0, 5, 0.05), 0, sigma_classic)
    fig, ax = plt.subplots(2)
    ax[0].scatter(np.arange(0, 5, 0.05), y_list[0], c="red", label="x(t)")
    ax[1].scatter(np.arange(0, 5, 0.05), y_list[1], c="black", label="x'(t)")
    for a in ax:
        a.legend()
    plt.show()

    y_list = runge2([first, second], [1, 0], np.arange(0, 5, 0.05), 1, sigma_38)
    fig, ax = plt.subplots(2)
    ax[0].scatter(np.arange(0, 5, 0.05), y_list[0], c="red", label="x(t)")
    ax[1].scatter(np.arange(0, 5, 0.05), y_list[1], c="black", label="x'(t)")
    for a in ax:
        a.legend()
    plt.show()

    print("Plots printed")


def task_2():
    print("Task 2", "*" * 100)
    print("Euler", "_" * 100)
    h = 0.01
    x_range = np.arange(0, 30, h)
    y_list = euler([first_t, second_t], [3, 0], x_range, 0.5)
    fig, ax = plt.subplots(2)
    print(len(y_list[1]), " ", len(y_list[0]))
    ax[0].plot(y_list[0], y_list[1], c="red", label="x'(x)")

    ax[1].scatter(x_range, y_list[1], c="black", label="x'(t)")
    for a in ax:
        a.legend()
    plt.show()
    print("Plot printed")

    print("Runge", "_" * 100)
    h = 0.01
    sigma_38 = np.array([1 / 8, 3 / 8, 3 / 8, 1 / 8])
    x_range = np.arange(0, 30, h)
    y_list = runge([first_t, second_t], [3, 0], x_range, 1, sigma_38, 0.5)
    fig, ax = plt.subplots(2)
    print(len(y_list[1]), " ", len(y_list[0]))
    ax[0].plot(y_list[0], y_list[1], c="red", label="x'(x)")

    ax[1].scatter(x_range, y_list[1], c="black", label="x'(t)")
    for a in ax:
        a.legend()
    plt.show()
    print("Plot printed")


def task_3():
    print("Task 2", "*" * 100)
    print("Richardson", "_" * 100)
    x_range = [0, 5]
    sigma_38 = np.array([1 / 8, 3 / 8, 3 / 8, 1 / 8])
    x_range, y_list = richardson(0.001, [first_t, second_t], [1, 0], x_range, 0.5, sigma_38)
    fig, ax = plt.subplots(2)
    print(len(y_list[1]), " ", len(y_list[0]))
    ax[0].plot(x_range, list(map(np.cos, x_range)), 'b-', label="xe(t)")
    ax[0].plot(x_range, y_list[0], 'g-', label="x(t)")
    ax[1].scatter(x_range, y_list[1], c="black", label="x'(t)")
    # ax[2].plot(x_range, list(map(np.cos, x_range)), c="red", label="Точное решение")
    for a in ax:
        a.legend()
    plt.show()
    print("Plot printed")


def main():
    task_1()
    task_2()
    task_3()


if __name__ == "__main__":
    main()
