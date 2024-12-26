import matplotlib.pyplot as plt
import numpy as np

t = np.linspace(0, 0.02, 1000)
fm = 300
fc = 10*fm
signal = 2 * np.cos(2*np.pi*fm*t) * np.cos(2*np.pi*fc*t) - np.sin(2*np.pi*fm*t) * np.sin(2*np.pi*fc*t)

U = 2 * np.cos(2*np.pi*fm*t)
Q = np.sin(2*np.pi*fm*t)


plt.figure(figsize=(12, 7))

plt.subplot(1, 1, 1)
plt.plot(t, signal, label='ориг сигнал')
#plt.title('оригинальный сигнал')
plt.grid(True)

plt.subplot(1, 1, 1)
plt.plot(t, np.sqrt(U ** 2 + Q ** 2), label='огибающая', color = "r")
#plt.title('огибающая')
plt.grid(True)

plt.subplot(1, 1, 1)
plt.plot(t, np.arctan2(Q, U), label='arctan', color = "orange")
#plt.title('огибающая')
plt.grid(True)

plt.tight_layout()
plt.show()