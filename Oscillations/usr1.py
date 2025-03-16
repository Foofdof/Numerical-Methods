import numpy as np
import matplotlib.pyplot as plt
from GirMethod.methods import gear_method


def oscillator_equations(t, y, delta, w0, alpha, beta, A, w):
    x, v = y
    dxdt = v
    dvdt = -2 * delta * v - w0 ** 2 * x - alpha * x ** 2 - beta * x ** 3 + A * np.sin(w * t)
    return np.array([dxdt, dvdt])


def rk4_solve(f, y0, t_array, args=()):
    """
    Решение системы ОДУ методом Рунге–Кутты 4-го порядка с фиксированным шагом.
    :param f:   функция f(t, y, *args), возвращающая правую часть ОДУ
    :param y0:  начальные условия (массив/список)
    :param t_array: массив временных точек (равномерная сетка)
    :param args: дополнительные параметры, передаваемые в f
    :return: массив размером (len(t_array), len(y0)) с решениями для каждого t
    """
    n = len(t_array)
    dt = t_array[1] - t_array[0]
    sol = np.zeros((n, len(y0)))
    sol[0, :] = y0

    for i in range(n - 1):
        t = t_array[i]
        y = sol[i, :]

        k1 = f(t, y, *args)
        k2 = f(t + dt / 2, y + dt / 2 * k1, *args)
        k3 = f(t + dt / 2, y + dt / 2 * k2, *args)
        k4 = f(t + dt, y + dt * k3, *args)

        sol[i + 1, :] = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return sol


# ----------------- ПАРАМЕТРЫ -----------------

x0 = 0.3
v0 = 0.0

delta = 0.02  # коэффициенеат затухания
w0 = 1.0  # собственная (фундаментальная) частота системы
alpha = 0  # коэффициент квадратичной нелинейности
beta = 0  # коэффициент кубической нелинейности
A = 0.3  # амплитуда внешней силы
w = 2 # частота внешней силы

t_max = 400
dt = 0.05
t_array = np.arange(0, t_max, dt)

y0 = [x0, v0]

funcs = np.array([
    lambda var_list: var_list[2],
    lambda var_list: -2 * delta * var_list[2] - w0 ** 2 * var_list[1] - alpha * var_list[1] ** 2 - beta * var_list[1] ** 3 + A * np.sin(w * var_list[0])
])
initial_conditions = np.array([x0, v0], dtype=np.float64)
sigma_classic = np.array([1 / 6, 2 / 6, 2 / 6, 1 / 6])

# ----------------- ЧИСЛЕННОЕ РЕШЕНИЕ -----------------

# sol = rk4_solve(oscillator_equations, y0, t_array, args=(delta, w0, alpha, beta, A, w))
sol = gear_method(
        funcs=funcs,
        init_cond=initial_conditions,
        t_grid=t_array,
        base_method_c='ie',
        m=4,
    )

# sol[:, 0] = x(t), sol[:, 1] = v(t)
# x_t = sol[:, 0]
# v_t = sol[:, 1]

x_t = sol[0, :]
v_t = sol[1, :]

# ----------------- ПОСТРОЕНИЕ ГРАФИКОВ -----------------

plt.figure(figsize=(12, 12))

# 1) График x(t)
plt.subplot(3, 2, (1, 2))
plt.plot(t_array, x_t, 'r', label='Нелинейная система')
# plt.plot(t_array, x0*np.exp(-delta*t_array)*np.cos(w0*t_array), 'b', label='Линейная система')
plt.title('Координата x(t)')
plt.xlabel('t')
plt.ylabel('x')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')

# 2) График v(t)
plt.subplot(3, 2, (3, 4))
plt.plot(t_array, v_t, 'r')
plt.title('Скорость v(t)')
plt.xlabel('t')
plt.ylabel('v')
plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')

# 3) Фазовый портрет (x, v)
plt.subplot(3, 2, 5)
plt.plot(x_t, v_t, 'g')
plt.title('Фазовый портрет')
plt.xlabel('x')
plt.ylabel('v')
plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')

# 4) Спектр Фурье x(t)
X_f = np.fft.fft(x_t)
freqs = np.fft.fftfreq(len(t_array), d=dt)
angular_freqs = 2 * np.pi * freqs

# Берём только строго положительные частоты для наглядности
idx = np.where(angular_freqs > 0)
angular_freqs_pos = angular_freqs[idx]
X_f_pos = X_f[idx]

# Находим все пики спектра (используем модуль спектра)
from scipy.signal import find_peaks
peaks, _ = find_peaks(np.abs(X_f_pos))

plt.subplot(3, 2, 6)
plt.plot(angular_freqs_pos[:(peaks[-1]+20)], np.abs(X_f_pos[:(peaks[-1]+20)]), 'm')
# plt.title('Фурье-спектр x(t)')
plt.xlabel('Угловая частота (рад/с)')
plt.ylabel('|X(f)|')
plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')

# Добавляем аннотацию для каждого пика спектра
for peak in peaks:
    peak_freq = angular_freqs_pos[peak]
    peak_amp = np.abs(X_f_pos[peak])
    plt.annotate(f'{peak_freq:.2f} rad/s', xy=(peak_freq, peak_amp),
                 xytext=(peak_freq, peak_amp*1.1),
                 arrowprops=dict(facecolor='black', arrowstyle='->'),
                 horizontalalignment='center')


plt.tight_layout()
plt.show()
