import numpy as np
import matplotlib.pyplot as plt

# Параметры OFDM
num_subcarriers = 64
cp_length = 16  # Длина циклического префикса
pilot_interval = 8 # Расстояние между пилотами

# QPSK модуляция
def qpsk_modulate(bits):
    return np.exp(1j * np.pi / 2 * bits[::2] + 1j * np.pi / 4 * (1 - 2 * bits[1::2]))

# Вставка гребенчатых пилотов
def insert_pilots(data):
    num_pilots = num_subcarriers // pilot_interval
    pilot_values = np.ones(num_pilots)  # Пилоты = 1
    piloted_data = np.zeros(num_subcarriers, dtype=complex)
    pilot_indices = np.arange(0, num_subcarriers, pilot_interval)
    data_indices = np.delete(np.arange(num_subcarriers), pilot_indices)
    piloted_data[pilot_indices] = pilot_values
    piloted_data[data_indices] = data
    return piloted_data

# Генерация случайных бит
num_bits = 2 * (num_subcarriers - num_subcarriers // pilot_interval) # 2 бита на символ для QPSK, вычитаем пилоты
bits = np.random.randint(0, 2, num_bits)

# Модуляция
modulated_data = qpsk_modulate(bits)

# Вставка пилотов
piloted_data = insert_pilots(modulated_data)

# IFFT
ofdm_symbol = np.fft.ifft(piloted_data, num_subcarriers)

# Добавление циклического префикса
ofdm_symbol_with_cp = np.concatenate((ofdm_symbol[-cp_length:], ofdm_symbol))

# Построение графиков
plt.figure(figsize=(12, 6))

# Спектр OFDM сигнала
plt.subplot(1, 2, 1)
plt.plot(np.abs(np.fft.fft(ofdm_symbol_with_cp)))
plt.title('OFDM Spectrum')
plt.xlabel('Frequency Bin')
plt.ylabel('Magnitude')

# Созвездие QPSK (до вставки пилотов)
plt.subplot(1, 2, 2)
plt.plot(np.real(modulated_data), np.imag(modulated_data), 'bo')
plt.title('QPSK Constellation')
plt.xlabel('In-Phase')
plt.ylabel('Quadrature')
plt.grid(True)

plt.tight_layout()
plt.show()