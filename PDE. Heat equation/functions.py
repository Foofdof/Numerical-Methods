import numpy as np


def make_matrix(funcs, region_list):
    """
    :param funcs [0] -- initial, [1] -- first bound, [2] -- second
    :param region_list:
    :return:
    """
    n = len(region_list[1])
    m = len(region_list[0])
    mtr = np.zeros((n, m))
    # initial conditions
    mtr[0, :] = funcs[0](region_list[0])
    # boundary conditions
    mtr[:, 0] = funcs[1](region_list[1])
    mtr[:, m - 1] = funcs[2](region_list[1])
    return mtr
