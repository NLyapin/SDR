import numpy as np
import matplotlib.pyplot as plt

from numpy import fft, argmax, arange, abs
from numpy import linspace, real, imag, log10, absolute, append, fliplr, matmul, pi, zeros, arange, cos, exp
from sk_dsp_comm.sigsys import my_psd
from scipy.signal import remez, freqz
from matplotlib.pyplot import subplots


def maximum_frequency(rx_nth_power, fs):
    """
    Calculate the frequency corresponding to the maximum amplitude in a signal's power spectral density.

    This function takes the Nth power received signal 'rxNthPower' and its sampling frequency 'fs', computes
    magnitude of the signal, and determines the frequency corresponding to the
    maximum amplitude.

    Args:
        rxNthPower (numpy.ndarray): Nth power received signal.
        fs (float): Sampling frequency of the signal.

    Returns:
        max_freq(float): The frequency corresponding to the maximum amplitude.

    Note:
        This function uses the Fast Fourier Transform (FFT) to calculate magnitude of
        the n powered received signal. The frequency corresponding to the maximum amplitude is determined
        .

    Example:
        receivedSignalSquared = np.array([...])  # Squared received signal
        samplingFrequency = 1000                 # Sampling frequency of the signal
        maxFrequency = maxFreq(receivedSignalSquared, samplingFrequency)
        # Returns the frequency corresponding to the maximum amplitude.
    """
    psd = fft.fftshift(abs(fft.fft(rx_nth_power)))
    f = arange(-fs / 2.0, fs / 2.0, fs / len(psd))
    max_freq = f[argmax(psd)]
    return max_freq

def plot_spectrum(x, Ts):
    """
    Plot the time-domain signal and its power spectral density (PSD).

    This function takes a complex input signal 'x' and its sampling period 'Ts' and plots
    both the time-domain representation of the signal (real and imaginary parts) and its
    power spectral density (PSD).

    Args:
        x (numpy.ndarray): Input complex signal.
        Ts (float): Sampling period of the signal.

    Returns:
        None

    Note:
        This function utilizes the 'my_psd' function from the 'ss' module to compute the
        power spectral density (PSD) of the input signal 'x'. Make sure the 'ss' module
        is available and the 'my_psd' function is defined before using this function.


    """
    N = len(x)
    fs = 1 / Ts
    t = Ts * linspace(0, N, N, endpoint=False)
    P_r, f = my_psd(x, 2**12, fs)

    fig10, [axx0, axx1] = plt.subplots(2, 1, figsize=(12.8, 9.6))
    fig10.suptitle("Frequency and Time Domain Representation", fontsize=20)

    axx0.plot(t, (x.real), label='Real', color='r')
    axx0.plot(t, (x.imag), label='Imag', color='b')
    axx0.legend()
    axx0.set_xlabel('t (sec)', fontsize=16)
    axx0.set_ylabel('Amplitude', fontsize=16)
    axx1.plot(f, 10 * log10(P_r))
    axx1.set_xlabel('Frequency (Hz)', fontsize=16)
    axx1.set_ylabel('Magnitude', fontsize=16)
    axx1.set_ylim([-150, 0])
    axx1.grid(True)
    axx0.grid(True)
    fig10.tight_layout()

def plot_response(w, h, title):
    "Utility function to plot response functions"
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(w, 20 * log10(absolute(h)))
    ax.set_ylim(-150, 30)
    ax.grid(True)
    ax.set_xlabel('Frequency (Hz)', fontsize=16)
    ax.set_ylabel('Gain (dB)', fontsize=16)
    ax.set_title(title, fontsize=20)
    
def costas_loop_QAM(rx, fs, mu, estimated_frequency, theta_init,
                    display_output):
    r = rx
    f0 = estimated_frequency
    Ts = 1 / fs
    nyquist_freq = fs / 2
    N = len(rx)
    time = N * Ts
    t = arange(0, time, Ts)
    fl = 201
    cut_off_freq = 0.4 * nyquist_freq
    start_freq = 0.3 * nyquist_freq
    ff = [0, start_freq, cut_off_freq, nyquist_freq]
    fa = [1, 0]
    taps = remez(fl, ff, fa, fs=fs)
    wBP, hBP = freqz(taps, [1], worN=1048, fs=fs)

    taps = taps.reshape((1, len(taps)))
    theta = zeros(len(t))
    theta[0] = theta_init
    q = fl
    z1 = zeros(q)
    z2 = zeros(q)
    z3 = zeros(q)
    z4 = zeros(q)
    carrier_estimation = zeros(N)
    complex_exp_estimation = zeros(N, dtype=complex)

    for k in range(len(t) - 1):
        s = 2 * r[k]
        z1 = append(z1[1:len(z1) + 1], s * cos(2 * pi * f0 * t[k] + theta[k]))
        z2 = append(z2[1:len(z2) + 1],
                    s * cos(2 * pi * f0 * t[k] + pi / 4 + theta[k]))
        z3 = append(z3[1:len(z3) + 1],
                    s * cos(2 * pi * f0 * t[k] + pi / 2 + theta[k]))
        z4 = append(z4[1:len(z4) + 1],
                    s * cos(2 * pi * f0 * t[k] + 3 * pi / 4 + theta[k]))
        lpf1 = matmul(fliplr(taps), z1.reshape((len(z1), 1)))
        lpf2 = matmul(fliplr(taps), z2.reshape((len(z2), 1)))
        lpf3 = matmul(fliplr(taps), z3.reshape((len(z3), 1)))
        lpf4 = matmul(fliplr(taps), z4.reshape((len(z4), 1)))
        theta[k + 1] = (theta[k] +
                        mu * lpf1[0, 0] * lpf2[0, 0] * lpf3[0, 0] * lpf4[0, 0])

        carrier_estimation[k] = cos((2 * pi * f0 * t[k] + theta[k]))
        complex_exp_estimation[k] = exp(
            -1j * (2 * pi * estimated_frequency * t[k] + theta[k]))
    if display_output:
        plot_response(wBP, hBP, "LP Filter for Costas Loop")

        fig5, axs = plt.subplots(2, 1, figsize=(12.8, 9.6))
        fig5.suptitle("Costas Loop Phase Recovery", fontsize=20)
        axs[0].plot(t, theta)
        axs[0].set_title("Theta", fontsize=18)
        axs[0].set_ylabel("Phase Offset", fontsize=16)
        axs[0].set_xlabel("t (s)", fontsize=16)
        axs[1].plot(t, carrier_estimation)
        axs[1].set_title('Estimated Carrier', fontsize=18)
        axs[1].set_ylabel("Amplitude", fontsize=16)
        axs[1].set_xlabel("t (s)", fontsize=16)
        fig5.tight_layout()
    return carrier_estimation, theta, complex_exp_estimation

# Открываем файл для чтения
with open('../../build/single_adalm_rx.txt', 'r') as file:
    # Читаем все строки из файла
    lines = file.readlines()

# Инициализируем список для хранения данных
data = []
imag = []
real = []
count = []


# Обрабатываем каждую строку
counter  = 0
for line in lines:
    line = line.strip()  # Убираем лишние пробелы
    if line:  # Пропускаем пустые строки
        try:
            # Разделяем строку по запятой и преобразуем в числа
            numbers = list(map(float, [value.strip() for value in line.split(',')]))
            # Создаём комплексное число (I + jQ)
            complex_number = complex(numbers[0], numbers[1])
            data.append(complex_number)
            imag.append(numbers[1])
            real.append(numbers[0])
            count.append(counter)
            counter += 1
        except ValueError:
            print(f"Ошибка преобразования строки: {line}")

# Преобразуем список в массив NumPy
# rx_nothreads = np.array(data)
fig, axs = plt.subplots(2, 1, layout='constrained')
plt.figure(1)
axs[1].plot(count, np.abs(data),  color='grey')  # Используем scatter для диаграммы созвездия
axs[0].plot(count,(imag),color='red')  # Используем scatter для диаграммы созвездия
axs[0].plot(count,(real), color='blue')  # Используем scatter для диаграммы созвездия
plt.show()
# Вывод загруженных данных
# print(rx_nothreads)
start = 74000
end = 78000
filtered_i = real[start:end - 1]
filtered_q = imag[start:end - 1]
fig, axs = plt.subplots(2, 1, layout='constrained')
# plt.figure(1)
#axs[1].plot(count, np.abs(data),  color='grey')  # Используем scatter для диаграммы созвездия
axs[0].plot(count[start:end - 1],(filtered_q),color='red')  # Используем scatter для диаграммы созвездия
axs[0].plot(count[start:end - 1],(filtered_i), color='blue')  # Используем scatter для диаграммы созвездия

plt.show()

count_01 = 0
rx_complex = []

for i in range(len(filtered_i)):
    rx_complex.append(complex(filtered_i[i], filtered_q[i]))

rx = np.array(rx_complex) / 2**11
x = rx.real 
y = rx.imag 
# plot the complex numbers 
fig, axs = plt.subplots(2, 1, layout='constrained')
axs[0].plot(x, y, 'g*') 
print(len(rx))
# # COARSE FREQUENCY CORRECTION
sampling_rate = int(10e6)
Ts = 1 / sampling_rate
t = np.arange(0, len(rx) * Ts, Ts)
r4th = rx**4
coarse_frequency = (maximum_frequency(r4th, sampling_rate) / 4)
print(f'coarse1 = {coarse_frequency}')
coarse_demodulator = np.exp(-1j * 2 * np.pi * coarse_frequency * t)
coarse_demodulated_baseband_signal = rx * coarse_demodulator
# plot_spectrum(coarse_demodulated_baseband_signal, Ts)
# plot_spectrum(r4th, Ts)
# plt.show() 
# x = coarse_demodulated_baseband_signal.real 
# y = coarse_demodulated_baseband_signal.imag 
# plt.plot(x, y, 'g*') 
# plt.ylabel('Imaginary') 
# plt.xlabel('Real') 
# plt.show()

# BAND SHIFTING BEFORE PHASE CORRECTION
fc = int(2e6)
shifted_before_CL = coarse_demodulated_baseband_signal * np.exp( 1j * 2 * np.pi * fc * t)
# plot_spectrum(shifted_before_CL, Ts)
carrier_est, theta, complex_exp_est = costas_loop_QAM(
    np.real(shifted_before_CL), sampling_rate, 0.2, fc, np.pi  / 6, False)
baseband_signal = complex_exp_est * shifted_before_CL
# plot_spectrum(baseband_signal, Ts)
# plt.show()

x = baseband_signal.real 
# extract imaginary part using numpy array 
y = baseband_signal.imag 
# plot the complex numbers 
axs[1].plot(x, y, 'g*') 


plt.show()