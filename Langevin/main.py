import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm
import seaborn as sns

x0 = 0
t0 = 0
T = 1
zeta = 1
D = 1


def external_force_1(x, t):
    return 1


def external_force_2(x, t):
    return -1*2*np.sin(2*np.pi*x)


def external_force_3(x, t):
    return -1*2*np.sin(2*np.pi*x)+1


def sol_langevin(dt=0.01, n=100, F_ext=None):
    N_steps = int(T / dt)
    t = np.linspace(t0, T, N_steps + 1)
    x = np.zeros((n, N_steps + 1))
    x[:, 0] = x0

    sqrt_2Ddt = np.sqrt(2 * D * dt)
    dW = np.random.randn(n, N_steps) * sqrt_2Ddt
    dt_over_zeta = dt / zeta

    if F_ext is not None:
        for i in range(N_steps):
            F = F_ext(x[:, i], t[i])
            x[:, i + 1] = x[:, i] + F * dt_over_zeta + dW[:, i]
    else:
        x[:, 1:] = x0 + np.cumsum(dW, axis=1)

    return x, t, dW, dt_over_zeta


def task_1(F_ext=None):
    x, t, dW, dtz = sol_langevin(dt=0.01, n=1000, F_ext=F_ext)
    mean_x = np.mean(x, axis=0)
    msd = np.mean((x - x0) ** 2, axis=0)

    # Plotting mean position vs time
    plt.figure(figsize=(8, 5))
    plt.plot(t, mean_x, c='black')
    plt.xlabel('Time t')
    plt.ylabel('Mean position ⟨x(t)⟩')
    plt.title('Mean Position by Time')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    plt.show()

    slope, intercept, r, p, se = stats.linregress(t, msd)
    D_estimated = slope / 2.0

    print(f"Fitted line: MSD = {slope:.4f} +- {se:.4f} * t")
    print(f"Estimated diffusion coefficient D = {D_estimated:.4f} +- {se / 2.:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(t, msd, label='Simulation')
    plt.plot(t, slope * t + intercept, color='black', linestyle='dashdot', label='Linear Fit')
    plt.xlabel('Time t')
    plt.ylabel('Mean Square Displacement ⟨(x(t) - x₀)²⟩')
    plt.title('Mean Square Displacement vs Time with Linear Fit')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    plt.show()

    moments_order = [1, 2, 3, 4]
    moments_simulation = {}
    for n in moments_order:
        moments_simulation[n] = np.mean((x-np.mean(x, axis=0)) ** n, axis=0)

    moments_theoretical = {}

    def double_factorial(k):
        if k <= 0:
            return 1
        else:
            return k * double_factorial(k - 2)

    print(double_factorial(5))

    for k in moments_order:
        if k % 2 == 0:
            moments_theoretical[k] = (2 * D * t) ** (k/2.) * double_factorial(k - 1)
        else:
            moments_theoretical[k] = np.zeros_like(t)

    plt.figure(figsize=(12, 10))

    for i, n in enumerate(moments_order, 1):
        plt.subplot(2, 2, i)
        plt.plot(t, moments_simulation[n], color='black', linestyle='dashdot', label=f'Simulation ⟨x(t)^{n}⟩')
        plt.plot(t, moments_theoretical[n], color='red', label=f'Theory ⟨x(t)^{n}⟩')
        plt.xlabel('Time t')
        plt.ylabel(f'⟨x(t)^{n}⟩')
        plt.title(f'Moment of {n} order')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()

    selected_times_indices = [10, 30, 50, 100]
    selected_times = t[selected_times_indices]

    plt.figure(figsize=(12, 10))

    for i, (idx, time_point) in enumerate(zip(selected_times_indices, selected_times), 1):
        plt.subplot(2, 2, i)

        x_t = x[:, idx]

        sns.histplot(x_t, bins=80, stat='density', color='pink', label='Simulation', kde=False, edgecolor='none')
        x_th_range = np.linspace(x_t.min(), x_t.max(), 1000)
        if F_ext is not None:
            P_x_t = norm.pdf(x_th_range, loc=F_ext(x_th_range, time_point)*time_point/zeta, scale=np.sqrt(2 * D * time_point))
        else:
            P_x_t = norm.pdf(x_th_range, loc=0, scale=np.sqrt(2 * D * time_point))


        plt.plot(x_th_range, P_x_t, color='black', linestyle='dashdot', label='Theory')

        plt.xlabel('x(t)')
        plt.ylabel('Probability density P(x, t)')
        plt.title(f'Coordinate distribution at time moment t = {time_point:.2f}')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()


def task_2():

    for n_trial in range(10, 110, 10):
        x, t, dW = sol_langevin(n=n_trial)
        mean_x = np.mean(x, axis=0)
        msd = np.mean((x - x0) ** 2, axis=0)

        moments_order = [1, 2, 3, 4]
        moments_simulation = {}
        for n in moments_order:
            moments_simulation[n] = np.mean((x-np.mean(x, axis=0)) ** n, axis=0)

        moments_theoretical = {}

        def double_factorial(k):
            if k <= 0:
                return 1
            else:
                return k * double_factorial(k - 2)
        for k in moments_order:
            if k % 2 == 0:
                moments_theoretical[k] = (2 * D * t) ** (k/2.) * double_factorial(k - 1)
            else:
                moments_theoretical[k] = np.zeros_like(t)

        plt.figure(figsize=(12, 10))

        for i, n in enumerate(moments_order, 1):
            plt.subplot(2, 2, i)
            plt.plot(t, moments_simulation[n], 'b-', label=f'Simulation ⟨x(t)^{n}⟩')
            plt.plot(t, moments_theoretical[n], 'r--', label=f'Theory ⟨x(t)^{n}⟩')
            plt.xlabel('Time t')
            plt.ylabel(f'⟨x(t)^{n}⟩')
            plt.title(f'Moment of {n} order ({n_trial})')
            plt.legend()
            plt.grid(True)

        print(n_trial)
        print(np.sum(moments_simulation[1]*0.01, axis=0))
        print(np.sum(moments_simulation[3]*0.01, axis=0))
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    task_1(None)
    # task_1(external_force_1)
    # task_1(external_force_2)
    # task_1(external_force_3)
    # task_2()
