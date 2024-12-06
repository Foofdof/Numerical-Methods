import methods
import numpy as np
import matplotlib.pyplot as plt


def task1():
    h = 0.05
    tau = 1e-5
    k = 1

    x = np.linspace(0, 1, int((1 - 0) / h))
    y = np.linspace(0, 1, int((1 - 0) / (h)))
    t_grid = np.linspace(0, 0.1, int((0.1 - 0) / tau))

    funcs = [lambda var_list: 0]
    init_cond = [lambda x_vec: 1+np.sin(np.pi*x_vec)]
    bound_cond = [lambda t: np.cos(t), lambda t: np.cos(t)]
    funcs2d = [lambda var_list: 0]
    init_cond2d = [lambda x_vec: 1 + np.sin(np.pi * x_vec)]
    bound_cond2d = [[lambda t: np.cos(t), lambda t: np.cos(t)],
                    [lambda t: np.cos(t), lambda t: np.cos(t)]]

    sol_expl = methods.explicit_scheme(funcs, init_cond, bound_cond, 1, t_grid, x)
    sol_impl = methods.implicit_scheme(funcs, init_cond, bound_cond, 1, t_grid, x)
    # sol_nick = methods.weighted_impl_scheme(funcs, init_cond, bound_cond, 1, t_grid, x)
    # # sol_rich = methods.richardson(funcs, init_cond, bound_cond, 1, t_grid, x)
    # sol_df = methods.dufort_franc(funcs, init_cond, bound_cond, 1, t_grid, x)

    sol_expl2d = methods.explicit_scheme_ND(funcs2d, init_cond2d, bound_cond2d, k, t_grid, (x, y))
    sol_adi = methods.adi_solver(funcs2d, init_cond2d, bound_cond2d, k, t_grid, (x, y))

    xl, tl = np.meshgrid(x, t_grid)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(tl, xl, sol_expl-sol_impl, cmap='cividis')
    # ax.plot_surface(tl, xl, sol_impl, cmap='cividis')
    # ax.plot_surface(tl, xl, sol_nick, cmap='cividis')
    # ax.plot_surface(tl, xl, sol_df, cmap='cividis')
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_zlabel('U')
    plt.show()

    indices = np.linspace(0, len(t_grid) - 1, 4, dtype=int)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    vmin = np.min(sol_expl2d)
    vmax = np.max(sol_expl2d)

    for i, idx in enumerate(indices):
        im = axes[i].imshow(
            sol_expl2d[idx], extent=[0, 1, 0, 1], origin='lower', cmap='hot', vmin=vmin, vmax=vmax
        )
        axes[i].set_title(f'Температура в t={t_grid[idx]:.4f} сек')
        axes[i].set_xlabel('x')
        axes[i].set_ylabel('y')
        fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04, label='Температура')

    plt.tight_layout()
    plt.show()


    indices = np.linspace(0, len(t_grid) - 1, 4, dtype=int)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    vmin = np.min(sol_expl2d)
    vmax = np.max(sol_expl2d)

    for i, idx in enumerate(indices):
        im = axes[i].imshow(
            sol_adi[idx], extent=[0, 1, 0, 1], origin='lower', cmap='hot', vmin=vmin, vmax=vmax
        )
        axes[i].set_title(f'Температура в t={t_grid[idx]:.4f} сек')
        axes[i].set_xlabel('x')
        axes[i].set_ylabel('y')
        fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04, label='Температура')

    plt.tight_layout()
    plt.show()


def task2():
    h = 0.05
    tau = 1e-4
    k = 1
    c = 1

    x = np.linspace(0, 1, int((1 - 0) / h))
    t_grid = np.linspace(0, 0.1, int((0.1 - 0) / tau))

    funcs = [lambda var_list: 0]
    init_cond = [lambda x_vec: 100*x_vec]
    bound_cond = [lambda t: 0, lambda t: 100]

    sol_expl = methods.diff_conv_expl_scheme(funcs, init_cond, bound_cond, k, c, t_grid, x)
    sol_impl = methods.diff_conv_expl_scheme(funcs, init_cond, bound_cond, k, c, t_grid, x)

    xl, tl = np.meshgrid(x, t_grid)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(tl, xl, sol_expl, cmap='cividis')
    ax.plot_surface(tl, xl, sol_impl, cmap='cividis')
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_zlabel('U')
    plt.show()


if __name__ == '__main__':
    task2()
