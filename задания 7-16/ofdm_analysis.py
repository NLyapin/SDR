import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate

num_subcarriers = 64
cp_length = 16  
pilot_interval = 8  

def qpsk_modulate(bits):
    return np.exp(1j * np.pi / 2 * bits[::2] + 1j * np.pi / 4 * (1 - 2 * bits[1::2]))

def insert_pilots(data):
    num_pilots = num_subcarriers // pilot_interval
    pilot_values = np.ones(num_pilots)  
    piloted_data = np.zeros(num_subcarriers, dtype=complex)
    pilot_indices = np.arange(0, num_subcarriers, pilot_interval)
    data_indices = np.delete(np.arange(num_subcarriers), pilot_indices)
    piloted_data[pilot_indices] = pilot_values
    piloted_data[data_indices] = data
    return piloted_data

def cross_correlation_plot(signal, reference, ax):
    """График кросс-корреляции."""
    correlation = correlate(signal, reference, mode='full')
    
    ax.plot(np.abs(correlation))
    ax.set_title('Кросс-корреляция')
    ax.set_xlabel('Отсчеты')
    ax.set_ylabel('Модуль')

def time_domain_plot(signal, title, ax):
    """График сигнала во временной области."""
    
    ax.plot(np.real(signal), label='Действ')
    ax.plot(np.imag(signal), label='Мним')
    ax.set_title(title)
    ax.set_xlabel('Отсчеты')
    ax.set_ylabel('Амплитуда')
    ax.legend()

def channel_estimation_plot(channel_response, ax): 
    """График оценки канала."""
    
    
    ax.plot(np.abs(channel_response), 'o-') 
    ax.set_title('Оценка канала')
    ax.set_xlabel('Номер поднесущей') 
    ax.set_ylabel('Модуль')
    ax.grid(True) 

def constellation_plot(data, title, ax):
    """График созвездия."""
    
    ax.plot(np.real(data), np.imag(data), 'bo')
    ax.set_title(title)
    ax.set_xlabel('Действительная часть')
    ax.set_ylabel('Мнимая часть')
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box') 




num_bits = 2 * (num_subcarriers - num_subcarriers // pilot_interval)  
bits = np.random.randint(0, 2, num_bits)


modulated_data = qpsk_modulate(bits)


piloted_data = insert_pilots(modulated_data)


ofdm_symbol = np.fft.ifft(piloted_data, num_subcarriers)


ofdm_symbol_with_cp = np.concatenate((ofdm_symbol[-cp_length:], ofdm_symbol))



channel_response = [0.8, 0.2+0.1j]  
received_signal = np.convolve(ofdm_symbol_with_cp, channel_response, mode='same')

noise_power = 0.01
noise = np.sqrt(noise_power/2)*(np.random.randn(len(received_signal)) + 1j*np.random.randn(len(received_signal)))
received_signal += noise




received_signal_no_cp = received_signal[cp_length:]


received_fft = np.fft.fft(received_signal_no_cp, num_subcarriers)


pilot_indices = np.arange(0, num_subcarriers, pilot_interval)
received_pilots = received_fft[pilot_indices]
transmitted_pilots = np.ones(len(pilot_indices)) 

channel_response = received_pilots / transmitted_pilots


interpolated_channel_response = np.interp(np.arange(num_subcarriers), pilot_indices, channel_response)
equalized_data = received_fft / interpolated_channel_response


data_indices = np.delete(np.arange(num_subcarriers), pilot_indices)
equalized_data_no_pilots = equalized_data[data_indices]


fig, axes = plt.subplots(2, 3, figsize=(12, 6)) 


cross_correlation_plot(received_signal, ofdm_symbol_with_cp[:cp_length], axes[0, 0])


time_domain_plot(ofdm_symbol, 'OFDM Сигнал', axes[0, 1])
time_domain_plot(ofdm_symbol_with_cp, 'OFDM Сигнал с ЦП', axes[0, 2])


constellation_plot(modulated_data, 'Созвездие до эквализации', axes[1, 0])


channel_estimation_plot(channel_response, axes[1, 1])


constellation_plot(equalized_data_no_pilots, 'Созвездие после эквализации', axes[1, 2])

plt.tight_layout() 
plt.show()