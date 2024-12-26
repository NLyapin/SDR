import matplotlib.pyplot as plt
import numpy as np 
import adi

def word_to_bin(text):
    return ' '.join(format(ord(char), '08b') for char in text)

# def bin_to_word(bin_text):
#     bin_values = bin_text.split()
#     ascii_chars = [chr(int(b, 2)) for b in bin_values]
#     return ''.join(ascii_chars)

sdr = adi.Pluto('ip:192.168.2.4')
sdr.rx_lo = int(700e6)
sdr.tx_lo = int(700e6)
sdr.sample_rate = int(2.5e6)
sdr.rx_buffer_size = 10000
sdr.tx_cyclic_buffer = True
sdr._tx_buffer_size = 10000
sdr._rx_buffer_size = 10000


array = []

word = "Hello"
word_bin = word_to_bin(word).replace(' ', '')
arr = []
for i in range(0,len(word_bin)):
    arr.append((int(word_bin[i])) * 4000)

print(arr)
print(word_bin)
arr1 = []

for i in range(0, len(arr)):
    for j in range(0, 40):
        arr1.append(arr[i])

print(arr1)

for i in range(1024):
    if i < 300 or i > 700:
        array.append(complex(0))
    else:
        array.append(complex(4000))

plt.figure(figsize=(10, 4))
plt.plot(array)
plt.show()

array_rx = []
sdr.tx(array)
for i in range(160):
    rx_raw = sdr.rx()
    for i in rx_raw:
        array_rx.append(abs(i))

plt.figure(figsize=(10, 4))
plt.plot(array_rx)
plt.show()

# print(array)
# print("-------------------------------")
# print(array_rx)