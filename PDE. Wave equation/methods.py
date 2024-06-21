from functions import np, trian_mtr_al


def explicit_scheme(mtr1, a, h, tau, f):
    alpha = (tau * a / h) ** 2
    for i in range(1, len(mtr1) - 1):
        for j in range(1, len(mtr1[0]) - 1):
            mtr1[i + 1][j] = (alpha * (mtr1[i][j + 1] + mtr1[i][j - 1]) + 2 * mtr1[i, j] * (1 - alpha) - mtr1[i - 1][j]
                              + tau ** 2 * f[i][j])
    return mtr1


def implicit_scheme(mtr, a, h, tau, f, region_list, funcs, w):
    alpha = (tau * a / h) ** 2
    n = len(region_list[0])  # x
    m = len(region_list[1])  # t
    for s in range(1, m - 1):
        # 0 - a, 1 - c, 2 - b, 3 - f
        lists = np.zeros((4, n), dtype=np.float64)
        # c, others = 0
        lists[1, 0] = 1
        lists[1, -1] = 1
        # f
        lists[3, 0] = funcs[2](region_list[1][s + 1])
        lists[3, -1] = funcs[3](region_list[1][s + 1])
        for i in range(1, n - 1):
            lists[0, i] = w * alpha
            lists[1, i] = (1 + 2 * w * alpha)
            lists[2, i] = w * alpha
            lists[3, i] = 2 * mtr[s, i] - mtr[s - 1, i] + alpha * (1 - 2 * w) * (
                    mtr[s, i + 1] - 2 * mtr[s, i] + mtr[s, i - 1]) + alpha * w * (
                                  mtr[s - 1, i + 1] - 2 * mtr[s - 1, i] + mtr[s - 1, i - 1]) + (tau**2) * f[s, i]
        u_sn = trian_mtr_al(lists)
        mtr[s + 1] = np.copy(u_sn)
    return mtr
