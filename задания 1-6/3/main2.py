import numpy as np
import matplotlib.pyplot as plt

T = 2
tau = 1
N = 6
K = 1000

# Вычисление коэффициентов ряда Фурье 
def calc_fourier_coeffs(signal, period, n_harmonics):
    an = np.zeros(n_harmonics)
    bn = np.zeros(n_harmonics)
    for n in range(n_harmonics):
        an[n] = (2 / period) * np.trapz(signal * np.cos(2 * np.pi * n * t / period), t)
        bn[n] = (2 / period) * np.trapz(signal * np.sin(2 * np.pi * n * t / period), t)
    return an, bn

t = np.linspace(-T / 2, T / 2, K, endpoint=False)

# Формирование прямоугольного сигнала 
signal = np.zeros_like(t)
signal[(t >= -tau / 2) & (t < tau / 2)] = 1

# Вычисление коэффициентов ряда Фурье
an, bn = calc_fourier_coeffs(signal, T, N + 1) 

# Вычисление амплитуд и фаз гармоник
An = np.sqrt(an**2 + bn**2)
phin = np.arctan2(bn, an)

print("Коэффициенты an:", an)
print("Коэффициенты bn:", bn)
print("Амплитуды An:", An)
print("Фазы phin (рад):", phin)

fig, axs = plt.subplots(2, 2, figsize=(14, 8))

# Исходный сигнал
axs[0, 0].plot(t, signal)
axs[0, 0].set_title('Прямоугольный сигнал')
axs[0, 0].set_xlabel('Время, с')
axs[0, 0].set_ylabel('Амплитуда')

# Спектр амплитуд
axs[0, 1].stem(np.arange(N + 1), An)
axs[0, 1].set_title('Спектр амплитуд')
axs[0, 1].set_xlabel('Номер гармоники')
axs[0, 1].set_ylabel('Амплитуда')

# Спектр фаз
axs[1, 0].stem(np.arange(N + 1), phin)
axs[1, 0].set_title('Спектр фаз')
axs[1, 0].set_xlabel('Номер гармоники')
axs[1, 0].set_ylabel('Фаза, рад')

# Синтез сигнала по нескольким гармоникам
for i, n_harmonics in enumerate([2, 4, 6]):
    synthesized_signal = np.zeros_like(t)
    for n in range(n_harmonics + 1): 
        synthesized_signal += An[n] * np.cos(2 * np.pi * n * t / T + phin[n])
    axs[1, 1].plot(t, synthesized_signal, label=f'{n_harmonics} гармоник')

axs[1, 1].set_title('Синтез сигнала')
axs[1, 1].set_xlabel('Время, с')
axs[1, 1].set_ylabel('Амплитуда')
axs[1, 1].legend()

plt.tight_layout()
plt.show()
