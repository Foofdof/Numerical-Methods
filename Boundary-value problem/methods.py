import numpy as np


def debc(g_list, x_range, funcs, h):
    x_list = np.arange(x_range[0], x_range[1], h)
    p_list = list(map(funcs[0], x_list))
    while h > 2 / max(p_list):
        h = h / 2
        x_list = np.arange(x_range[0], x_range[1], h)
        p_list = list(map(funcs[0], x_list))
    q_list = list(map(funcs[1], x_list))
    r_list = list(map(funcs[2], x_list))
    n = len(x_list)
    a_list = np.zeros(n)
    b_list = np.zeros(n)
    c_list = np.zeros(n)
    f_list = np.zeros(n)

    c_list[0] = (-2 * g_list[1] / (h ** 2) + g_list[0] * 2 / h + g_list[1] * q_list[0])
    b_list[0] = (-2 * g_list[1] / (h ** 2))
    f_list[0] = (g_list[1] * r_list[0] - (p_list[0] - 2 / h) * g_list[2])
    a_list[n - 1] = -2 * g_list[4] / (h ** 2)
    c_list[n - 1] = -2 * g_list[3] * (1 + p_list[n - 1]) / h + g_list[4] * (q_list[n - 1] - 2 / h ** 2)
    f_list[n - 1] = -2 * g_list[5] * (1 + p_list[n - 1]) / h + r_list[n - 1] * g_list[4]

    for i in range(1, n - 1):
        a_list[i] = (p_list[i] / 2 / h - 1 /(h ** 2))
        c_list[i] = (q_list[i] - 2 / (h ** 2))
        b_list[i] = (-p_list[i] / 2 / h - 1 / (h ** 2))
        f_list[i] = r_list[i]

    alpha = [b_list[0] / c_list[0]]
    beta = [f_list[0] / c_list[0]]
    for i in range(1, n):
        alpha.append(b_list[i] / (c_list[i] - alpha[i - 1] * a_list[i]))
        beta.append((f_list[i] + a_list[i] * beta[i - 1]) / (c_list[i] - a_list[i] * alpha[i - 1]))

    y_list = np.zeros(n)
    y_list[n - 1] = beta[n - 1]
    for i in range(n - 2, -1, -1):
        y_list[i] = alpha[i] * y_list[i + 1] + beta[i]

    return x_list, y_list
