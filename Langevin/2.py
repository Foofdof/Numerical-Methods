import numpy as np
import matplotlib.pyplot as plt

# Параметры модели
zeta = 1.0       # Коэффициент вязкости (безразмерный)
k_B = 1.0        # Постоянная Больцмана (безразмерная)
T = 1.0          # Температура (безразмерная)
Lambda = zeta * k_B * T  # Λ = ζ k_B T

# Коэффициент диффузии
D = k_B * T / zeta  # D = k_B T / ζ

# Обезразмерные параметры
t0 = 1.0          # Характеристическое время
x0 = np.sqrt(D * t0)  # Характеристическая длина

# Безразмерные параметры
D_dim = D
dt = 0.01         # Шаг по времени
N_steps = 1000    # Количество шагов
N_trajectories = 10000  # Количество траекторий

# Время
time = np.linspace(0, N_steps*dt, N_steps+1)

# Инициализация массива траекторий
x = np.zeros((N_trajectories, N_steps+1))

# Генерация случайных сил
# В безразмерной форме <ξ(t) ξ(t')> = 2 Λ δ(t - t')
# При численном моделировании, ∆t заменяет δ-функцию
# То есть, ξ(t) * sqrt(∆t) ~ N(0, 2 Λ ∆t)
# В безразмерной форме 2 Λ ∆t = 2 * zeta * k_B * T * dt / (x0^2 / t0)
# Поскольку x0^2 = D * t0 = k_B T / zeta * t0, получаем:
# 2 Λ ∆t = 2 * zeta * k_B * T * dt / (k_B * T / zeta * t0) = 2 * zeta^2 * dt / (k_B * T) * (k_B * T / zeta) = 2 * zeta * dt
# В безразмерной форме zeta =1, поэтому σ = sqrt(2 * dt)

sigma = np.sqrt(2 * dt)

# Генерация случайных приращений
dW = sigma * np.random.randn(N_trajectories, N_steps)

# Численное интегрирование уравнения Ланжевена
for i in range(N_steps):
    x[:, i+1] = x[:, i] + (1/zeta) * 0 * dt + (1/zeta) * dW[:, i]

# Среднее значение <x(t)>
x_mean = np.mean(x, axis=0)

# Средний квадрат смещения <x(t)^2>
x2_mean = np.mean(x**2, axis=0)

# Теоретическое значение среднего квадрата смещения
x2_theory = 2 * D_dim * time

# Оценка коэффициента пропорциональности путем линейной аппроксимации
# Используем только большие времена для аппроксимации
fit_start = int(0.5 * N_steps)
coeff, _ = np.polyfit(time[fit_start:], x2_mean[fit_start:], 1)
print(f"Оцененный коэффициент пропорциональности: {coeff:.4f}")
print(f"Теоретический коэффициент пропорциональности: {2 * D_dim:.4f}")

# Оценка коэффициента диффузии D
# D = (1/2) * slope
D_estimated = coeff / 2
print(f"Оцененный коэффициент диффузии D: {D_estimated:.4f}")
print(f"Теоретический коэффициент диффузии D: {D_dim:.4f}")

# Построение графиков
plt.figure(figsize=(10,6))
plt.plot(time, x2_mean, label='Численное <x(t)^2>')
plt.plot(time, x2_theory, 'r--', label='Теоретическое 2Dt')
plt.xlabel('Время (безразмерное)')
plt.ylabel('Средний квадрат смещения <x(t)^2>')
plt.title('Средний квадрат смещения частицы при свободной диффузии')
plt.legend()
plt.grid(True)
plt.show()
