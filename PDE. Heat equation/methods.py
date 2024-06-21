import numpy as np


def explicit_scheme(mtr1, a, h, tau, f):
    alpha = tau * a ** 2 / h ** 2
    for i in range(0, len(mtr1) - 1):
        for j in range(1, len(mtr1[0]) - 1):
            mtr1[i + 1][j] = alpha * (mtr1[i][j + 1] + mtr1[i][j - 1]) + (1 - 2 * alpha) * mtr1[i][j] + tau * f[i][j]
    return mtr1


def implicit_scheme(funcs, region_list, a, h, tau):
    alpha = tau * a ** 2 / h ** 2
    n = len(region_list[0])
    m = len(region_list[1])
    mtr = np.zeros((n, m))
    mtr[:, 0] = funcs[0](region_list[0])

    for j in range(m-1):
        k = np.zeros(n)
        l = np.zeros(n)
        k[0] = 0
        l[0] = funcs[1](region_list[1][j+1])
        for i in range(1, n):
            k[i] = - alpha / (alpha * k[i-1] - (1+2*alpha))
            l[i] = - (mtr[i, j] + alpha*l[i-1])/ (alpha * k[i-1] - (1+2*alpha))

        mtr[n-1, j+1] = funcs[2](region_list[1][j+1])
        for i in range(n-2, -1, -1):
            mtr[i][j+1] = k[i]*mtr[i+1][j+1] + l[i]

    return mtr.transpose()
