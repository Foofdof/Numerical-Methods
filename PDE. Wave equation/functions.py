import numpy as np


def make_matrix(funcs, region_list):
    """
    :param funcs [0],[1] -- initial 1,2; [2],[3] -- first, second bound
    :param region_list: [0] -- x_list, [1] -- t_list
    :return:
    """
    tau = region_list[1][1] - region_list[1][0]
    n = len(region_list[1])
    m = len(region_list[0])
    mtr = np.zeros((n, m), dtype=np.float64)
    # initial conditions
    mtr[0, :] = funcs[0](region_list[0])
    mtr[1, :] = mtr[0, :] + tau*funcs[1](region_list[0])
    # boundary conditions
    mtr[:, 0] = funcs[2](region_list[1])
    mtr[:, m - 1] = funcs[3](region_list[1])
    return mtr


def trian_mtr_al(lists):
    n = len(lists[0])
    a_list = lists[0]
    c_list = lists[1]
    b_list = lists[2]
    f_list = lists[3]

    alpha = [b_list[0] / c_list[0]]
    beta = [f_list[0] / c_list[0]]
    for i in range(1, n):
        alpha.append(b_list[i] / (c_list[i] - alpha[i - 1] * a_list[i]))
        beta.append((f_list[i] + a_list[i] * beta[i - 1]) / (c_list[i] - a_list[i] * alpha[i - 1]))

    y_list = np.zeros(n, dtype=np.float64)
    y_list[n - 1] = beta[n - 1]
    for i in range(n - 2, -1, -1):
        y_list[i] = alpha[i] * y_list[i + 1] + beta[i]

    return y_list
