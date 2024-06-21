import numpy as np
from functions import get_max


def pde_ziedel_solve(mtr1, h, l, eps):
    k = 1
    t = h ** 2 / l ** 2
    mtr2 = np.copy(mtr1)
    for i in range(1, len(mtr1) - 1):
        for j in range(1, len(mtr1[0]) - 1):
            mtr2[i][j] = (mtr1[i + 1][j] + mtr2[i - 1][j] + t * (mtr1[i][j + 1] + mtr2[i][j - 1])) / (2 + 2 * t)
    err = get_max(mtr2, mtr1)
    while abs(err) > eps:
        mtr1 = np.copy(mtr2)
        for i in range(1, len(mtr1) - 1):
            for j in range(1, len(mtr1[0]) - 1):
                mtr2[i][j] = (mtr1[i + 1][j] + mtr2[i - 1][j] + t * (mtr1[i][j + 1] + mtr2[i][j - 1])) / (2 + 2 * t)
        k = k + 1
        err = get_max(np.copy(mtr2), np.copy(mtr1))
    return mtr2, k


def pde_relax_solve(mtr1, h, l, eps, w):
    k = 1
    t = h ** 2 / l ** 2
    mtr2 = np.copy(mtr1)

    for i in range(1, len(mtr1) - 1):
        for j in range(1, len(mtr1[0]) - 1):
            try:
                # mtr2[i][j] = (mtr2[i + 1][j] + mtr1[i - 1][j] + t * (mtr2[i][j + 1] + mtr1[i][j - 1])) / (2 + 2 * t)
                ut = (mtr1[i + 1][j] + mtr2[i - 1][j] + t * (mtr1[i][j + 1] + mtr2[i][j - 1])) / (2 + 2 * t)
                mtr2[i][j] = mtr1[i][j] + w * (ut - mtr1[i][j])
            except RuntimeWarning:
                print(k, " ", i)
    # mtr2 = mtr1 + w * (mtr2 - mtr1)

    err = get_max(mtr2, mtr1)
    erp = err
    while eps < abs(err):
        mtr1 = np.copy(mtr2)
        for i in range(1, len(mtr1) - 1):
            for j in range(1, len(mtr1[0]) - 1):
                try:
                    # mtr2[i][j] = (mtr2[i + 1][j] + mtr1[i - 1][j] + t * (mtr2[i][j + 1] + mtr1[i][j - 1])) / (2 + 2 * t)
                    ut = (mtr1[i + 1][j] + mtr2[i - 1][j] + t * (mtr1[i][j + 1] + mtr2[i][j - 1])) / (2 + 2 * t)
                    mtr2[i][j] = mtr1[i][j] + w * (ut - mtr1[i][j])
                except RuntimeWarning:
                    print(k, " ", i)
        # mtr2 = np.copy(mtr1 + w*(mtr2 - mtr1))
        k = k + 1
        err = get_max(np.copy(mtr2), np.copy(mtr1))

    return mtr2, k
