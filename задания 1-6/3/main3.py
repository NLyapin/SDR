import numpy as np

T = 1 
f1 = 1 / T
K = 1000
t = np.linspace(0, T, K)

def calculate_integral(k, n, f1=f1, start=0, end=T):
    t_interval = np.linspace(start, end, K)  
    sk = np.sin(2 * np.pi * k * f1 * t_interval)
    sn = np.exp(1j * 2 * np.pi * n * f1 * t_interval)
    integral = np.trapz(sk * sn, t_interval)
    return integral

print("Проверка ортогональности для sin(kt) и exp(jnt):")
for k in range(1, 11):
    for n in range(-10, 11):
        integral = calculate_integral(k, n)
        print(f"k = {k}, n = {n}, Интеграл = {integral:.5f}")

f2 = 1.5 * f1 
print("\nПроверка ортогональности с измененной частотой:")
for k in range(1, 4):
    for n in range(-3, 4):
        integral = calculate_integral(k, n, f1=f2)
        print(f"k = {k}, n = {n}, Интеграл = {integral:.5f}")

start_time = 0.2 * T
end_time = 0.8 * T
print("\nПроверка ортогональности с измененным интервалом интегрирования:")
for k in range(1, 4):
    for n in range(-3, 4):
        integral = calculate_integral(k, n, start=start_time, end=end_time)
        print(f"k = {k}, n = {n}, Интеграл = {integral:.5f}")
