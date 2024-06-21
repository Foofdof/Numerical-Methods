import numpy as np


def make_matrix(funcs, region_list):
    n = len(region_list[0])
    m = len(region_list[1])
    mtr = np.zeros((n, m), dtype=np.float64)
    # y boundary condition
    mtr[:, 0] = funcs[2](region_list[0])
    mtr[:, -1] = funcs[3](region_list[0])
    # x boundary condition
    mtr[0, :] = funcs[0](region_list[1])
    mtr[-1, :] = funcs[1](region_list[1])
    return mtr


def get_max(m2, m1):
    return np.max(np.abs(m2 - m1))
