#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve, firwin
from dataclasses import dataclass
from typing import Optional, Tuple
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_raised_cosine_filter(sps: int, num_taps: int, beta: float) -> np.ndarray:
        """Создает FIR-фильтр с поднятым косинусом."""
        t = np.arange(num_taps) - (num_taps - 1) // 2
        h = np.sinc(t / sps) * np.cos(np.pi * beta * t / sps) / (1 - (2 * beta * t / sps)**2)
        # Обработка точек с делением на ноль
        singular_mask = np.isclose(1 - (2 * beta * t / sps)**2, 0, atol=1e-5)
        h[singular_mask] = (np.pi / 4) * np.sinc(1 / (2 * beta)) 
        return h / np.sum(h)  # Нормализация


@dataclass
class Config:
    filename: str
    start_sample: int
    end_sample: int
    sps: int
    algorithm: int
    filter_length: int
    rolloff: float
    num_eye_traces: int = 200
    plot_colors: Tuple[str, str] = ('#2E86C1', '#E74C3C')
    plot_alpha: float = 0.1

    def validate(self):
        if not 0 <= self.rolloff <= 1:
            raise ValueError("Rolloff factor must be between 0 and 1")
        if self.filter_length % 2 == 0:
            raise ValueError("Filter length must be odd")
        if self.algorithm not in (3, 4, 5):
            raise ValueError("Invalid algorithm selection")


class SignalProcessor:
    def __init__(self, config: Config):
        self.config = config
        self.iq_data = None
        self.filtered_iq = None
        self.synced_symbols = None
        self.errors = None
        self.offsets = None

    def load_data(self):
        try:
            self.iq_data = read_iq_data(
                self.config.filename,
                self.config.start_sample,
                self.config.end_sample
            )
            self.iq_data /= np.max(np.abs(self.iq_data)) + 1e-9
            logger.info("Data loaded successfully. Samples: %d", len(self.iq_data))
        except Exception as e:
            logger.error("Data loading failed: %s", e)
            raise

    def apply_filter(self):
        rc_filter = create_raised_cosine_filter(
            self.config.sps,
            self.config.filter_length,
            self.config.rolloff
        )
        self.filtered_iq = convolve(self.iq_data, rc_filter, mode='full')[:len(self.iq_data)]
        logger.debug("Applied raised cosine filter. Filter length: %d", len(rc_filter))

    def synchronize(self):
        try:
            self.synced_symbols, self.errors, self.offsets = gardner_timing_recovery(
                self.filtered_iq,
                self.config.sps,
                self.config.algorithm
            )
            logger.info("Timing recovery completed. Symbols recovered: %d", len(self.synced_symbols))
        except ValueError as e:
            logger.error("Synchronization failed: %s", e)
            raise

    def process(self):
        self.load_data()
        self.apply_filter()
        self.synchronize()

class Visualizer:
    def __init__(self, processor: SignalProcessor, config: Config):
        self.processor = processor
        self.config = config

    def create_all_figures(self):
        """Создает отдельные фигуры для каждого графика"""
        figures = []
        
        # Временные графики
        figures.append(self.create_time_series_figure(
            self.processor.iq_data,
            "Raw I/Q Data",
            (10, 4)
        ))
        
        figures.append(self.create_time_series_figure(
            self.processor.filtered_iq,
            "Filtered I/Q Data", 
            (10, 4)
        ))
        
        figures.append(self.create_error_plot(
            self.processor.errors,
            "Timing Error Detector Output",
            (8, 4)
        ))

        # Диаграммы созвездий
        figures.append(self.create_constellation_figure(
            self.processor.iq_data[::self.config.sps],
            "Raw Constellation",
            (6, 6)
        ))
        
        figures.append(self.create_constellation_figure(
            self.processor.filtered_iq[::self.config.sps],
            "Filtered Constellation",
            (6, 6)
        ))
        
        figures.append(self.create_constellation_figure(
            self.processor.synced_symbols,
            "Synchronized Constellation",
            (6, 6)
        ))

        # Глазковые диаграммы
        figures.append(self.create_eye_diagram_figure(
            self.processor.filtered_iq,
            self.config.sps,
            "Filtered Eye Diagram",
            (8, 4)
        ))
        
        figures.append(self.create_eye_diagram_figure(
            self.processor.synced_symbols,
            1,
            "Synced Eye Diagram", 
            (8, 4)
        ))

        # График смещений
        figures.append(self.create_offsets_plot(
            self.processor.offsets,
            "Timing Offsets",
            (8, 4)
        ))

        return figures

    def create_time_series_figure(self, data, title, figsize):
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(data.real, color=self.config.plot_colors[0], label='I')
        ax.plot(data.imag, color=self.config.plot_colors[1], label='Q')
        ax.set_title(title)
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def symbol_sync(y, N, Ng):
    """Символьная синхронизация с использованием циклического префикса."""
    N_samples = len(y)
    cc = np.zeros(N_samples, dtype=complex)
    threshold = 0.9

    # Добавляем искуственное смещение
    offset = 10
    y = np.concatenate((np.zeros(offset, dtype=complex), y))
    N_samples = len(y)

    for n in range(Ng, N_samples - N):
        W1 = y[n - Ng + 1:n + 1]
        W2 = y[n - N - Ng + 1:n - N + 1]
        
        numerator = np.sum(W1 * np.conj(W2))
        denominator = np.sqrt(np.sum(np.abs(W1)**2) * np.sum(np.abs(W2)**2))
        
        cc[n] = numerator / denominator if denominator != 0 else 0

        if np.abs(cc[n]) > threshold:
            m = n - N
            return m, cc
    return -1, cc  # Возвращаем -1, если синхронизация не найдена


def freq_sync(y, N, Ng):
    """Частотная синхронизация с использованием защитного интервала."""
    epsilon = np.angle(np.sum(np.conj(y[:Ng]) * y[N:N+Ng])) / (2 * np.pi)
    return epsilon

def estimate_channel(Yp, Xp):
    """Оценка канала по пилот-сигналам (метод наименьших квадратов)."""
    return Yp / Xp

def interpolate_channel(Hp, D, Nfft):
    """Линейная интерполяция частотной характеристики канала."""
    Hc = np.zeros(Nfft, dtype=complex)
    Np = len(Hp) # Number pilot

    for m in range(Np -1):
      Hc[m*D] = Hp[m]
      for d in range(1,D):
        Hc[m*D+d] = Hp[m] + (Hp[m+1]-Hp[m])*d/D
    
    Hc[(Np-1)*D:] = Hp[Np-1] # fill last values

    return Hc

def correct_signal(Yc, Hc):
    """Коррекция принятого сигнала."""
    return Yc / Hc

def generate_ofdm_signal(Nfft, Ng, Np, data_size):
    """Генерация тестового OFDM сигнала."""

    # Генерация случайных данных
    data = np.random.randint(0, 2, data_size) * 2 - 1  # BPSK модуляция

    # Размещение пилот-сигналов
    D = Nfft // Np
    pilot_values = np.ones(Np)  # Пилоты - единицы
    pilot_indices = np.arange(0, Nfft, D)
    data_indices = np.array(list(set(range(Nfft)) - set(pilot_indices)))

    X = np.zeros(Nfft, dtype=complex)
    X[pilot_indices] = pilot_values
    X[data_indices] = data

    # ОДПФ
    x = np.fft.ifft(X, Nfft)

    # Циклический префикс
    x_cp = np.concatenate((x[-Ng:], x))

    return x_cp, X, pilot_values, pilot_indices, data_indices

def apply_channel(x_cp, snr_db=20):
     """Применение канала с АБГШ и сдвигом"""
     # Добавление шума
     signal_power = np.mean(np.abs(x_cp)**2)
     noise_power = signal_power / (10**(snr_db / 10))
     noise = np.sqrt(noise_power / 2) * (np.random.randn(len(x_cp)) + 1j * np.random.randn(len(x_cp)))
     y = x_cp + noise
     return y

def ofdm_demodulate(y, Nfft, Ng, pilot_indices, data_indices):
    """Демодуляция OFDM сигнала."""
    
    # Удаление циклического префикса
    y_no_cp = y[Ng:]

    # ДПФ
    Y = np.fft.fft(y_no_cp, Nfft)

    # Выделение пилот-сигналов и данных
    Yp = Y[pilot_indices]
    Yc = Y[data_indices]
    
    return Y, Yp, Yc

def generate_funny_image(width, height, text):
    """Генерирует смешную картинку."""
    image = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("Arial.ttf", size=30)  # Укажите путь к вашему шрифту
    draw.text((50, 50), text, fill="black", font=font)
    # Добавьте здесь другие элементы для создания смешного эффекта, например, искажение изображения
    return image

if __name__ == "__main__":
    # Параметры OFDM
    Nfft = 64  # Размер ДПФ
    Ng = 16  # Длина защитного интервала
    Np = 8  # Количество пилот-сигналов
    data_size = Nfft - Np # data size

    # Генерация OFDM сигнала
    x_cp, X, pilot_values, pilot_indices, data_indices = generate_ofdm_signal(Nfft, Ng, Np, data_size)

    # Имитация канала (смещение и шум)
    y = apply_channel(x_cp)

    # Символьная синхронизация
    start_index, cc = symbol_sync(y, Nfft, Ng)
    print(f"Индекс начала символа: {start_index}")

    # Частотная синхронизация
    epsilon = freq_sync(y[start_index:], Nfft, Ng)
    print(f"Оценка частотного смещения: {epsilon}")

    # Компенсация CFO
    n = np.arange(len(y))
    y_corr_cfo = y * np.exp(-1j * 2 * np.pi * epsilon / Nfft * n)

    # Демодуляция
    Y, Yp, Yc = ofdm_demodulate(y_corr_cfo[start_index:], Nfft, Ng, pilot_indices, data_indices)

    # Оценка канала
    Hp = estimate_channel(Yp, pilot_values)

    # Интерполяция канала
    Hc = interpolate_channel(Hp, Nfft // Np, Nfft)

    # Коррекция сигнала
    Xc_eq = correct_signal(Yc, Hc[data_indices])

    # --- Генерация смешной картинки ---
    image = generate_funny_image(400, 200, "OFDM - это весело!")
    image.save("funny_ofdm_image.png")
    print("Смешная картинка сохранена как funny_ofdm_image.png")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.show()

def create_error_plot(self, errors, title, figsize):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(errors, color='#28B463', linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Symbol Index")
    ax.set_ylabel("Error")
    ax.grid(True, alpha=0.3)
    return fig

def create_constellation_figure(self, data, title, figsize):
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(data.real, data.imag, s=8, alpha=0.6,
                c=np.arange(len(data)), cmap='viridis')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("In-Phase")
    ax.set_ylabel("Quadrature")
    ax.set_aspect('equal')
    return fig

def create_eye_diagram_figure(self, signal, sps, title, figsize):
    fig, ax = plt.subplots(figsize=figsize)
    plot_eye_diagram(ax, signal, sps,
                    self.config.num_eye_traces,
                    title)
    return fig

def create_offsets_plot(self, offsets, title, figsize):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(offsets, color='#7D3C98', linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Symbol Index")
    ax.set_ylabel("Offset (samples)")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, self.config.sps)
    return fig
    
def read_iq_data(filename: str, start_sample: int, end_sample: int) -> np.ndarray:
    """Читает I/Q данные из бинарного файла."""
    if end_sample <= start_sample:
        raise ValueError("Invalid sample range")
    
    with open(filename, "rb") as f:
        f.seek(start_sample * 2)
        buffer = f.read((end_sample - start_sample) * 2)
    
    data = np.frombuffer(buffer, dtype=np.int16)
    return data[::2] + 1j * data[1::2]

def gardner_timing_recovery(signal: np.ndarray, sps: int, algorithm: int) -> Tuple[np.ndarray, ...]:
    """Улучшенная реализация с правильным расчетом смещений"""
    # Параметры петли
    damping = 0.707
    bandwidth = 0.01
    theta = bandwidth / (damping + 0.25/damping)
    kp = 2.7  # Коэффициент усиления детектора ошибок
    
    # Коэффициенты фильтра
    k1 = -4 * damping * theta / kp
    k2 = -4 * theta**2 / kp

    symbols = []
    errors = []
    offsets = []
    
    p1 = 0.0  # Интегратор первого порядка
    p2 = 0.0  # Интегратор второго порядка
    sample_index = 0

    while True:
        # Расчет индекса с учетом дробного смещения
        current_index = int(sample_index + round(p2 * sps))
        
        if current_index + 2*sps >= len(signal):
            break

        # Выборка данных
        current = signal[current_index]
        mid = signal[current_index + sps//2]
        next_ = signal[current_index + sps]

        # Расчет ошибки
        if algorithm == 3:
            error = (next_.real - current.real) * mid.real + (next_.imag - current.imag) * mid.imag
        elif algorithm == 4:
            error = np.real((np.conj(next_) - np.conj(current)) * mid)
        elif algorithm == 5:
            prev = signal[current_index - sps] if current_index >= sps else 0
            error = current.real*np.sign(prev.real) + current.imag*np.sign(prev.imag) - prev.real*np.sign(current.real) - prev.imag*np.sign(current.imag)
        else:
            raise ValueError("Invalid algorithm")

        # Обновление фильтра
        p1 += k1 * error + k2 * error
        p2 += p1
        
        # Нормализация смещения
        p2 = p2 % 1.0  # Дробная часть смещения

        symbols.append(current)
        errors.append(error)
        offsets.append(p2 * sps)  # Переводим в отсчеты
        
        sample_index += sps

    return np.array(symbols), np.array(errors), np.array(offsets)

def calculate_timing_error(signal: np.ndarray, idx: int, sps: int, algorithm: int) -> float:
    """Вычисляет ошибку синхронизации для разных алгоритмов."""
    current = signal[idx]
    if algorithm in (3, 4):
        mid = signal[idx + sps//2]
        next_ = signal[idx + sps]
        if algorithm == 3:
            return (next_.real - current.real) * mid.real + (next_.imag - current.imag) * mid.imag
        return np.real((np.conj(next_) - np.conj(current)) * mid)
    
    # M&M алгоритм
    prev = signal[idx - sps] if idx >= sps else 0
    return current.real * np.sign(prev.real) + current.imag * np.sign(prev.imag) - prev.real * np.sign(current.real) - prev.imag * np.sign(current.imag)

def plot_constellation(ax: plt.Axes, data: np.ndarray, title: str):
    """Визуализация диаграммы созвездия."""
    ax.scatter(data.real, data.imag, s=8, alpha=0.6, 
              c=np.arange(len(data)), cmap='viridis')
    ax.set_title(title, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("In-Phase")
    ax.set_ylabel("Quadrature")
    ax.set_aspect('equal')

def plot_eye_diagram(ax: plt.Axes, signal: np.ndarray, sps: int, num_traces: int, title: str):
    """Генерация глазковой диаграммы."""
    segment_length = 2 * sps
    for i in range(min(num_traces, len(signal)//sps)):
        seg = signal[i*sps : i*sps + segment_length].real
        ax.plot(seg, color='#34495E', alpha=0.15, linewidth=0.8)
    ax.set_title(title, fontsize=9)
    ax.set_xlim(0, segment_length-1)