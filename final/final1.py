import numpy as np
from scipy.linalg import eigh
import pandas as pd
import matplotlib.pyplot as plt
import os
# 設定 matplotlib 字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

# 題目參數設定
m1 = m2 = m3 = 8.46 * 10 **7  # kg
k1 = k2 = k3 = 71200000  # N/m
c1 = c2 = c3 = 1552226.752869  # Ns/m

# 質量矩陣 M (3x3)
M = np.diag([m1, m2, m3])

# 勁度矩陣 K (3x3)
K = np.array([
    [k1 + k2, -k2,       0],
    [-k2,     k2 + k3,  -k3],
    [0,      -k3,        k3]
])

# 阻尼矩陣 C (3x3)
C = np.array([
    [c1 + c2, -c2,       0],
    [-c2,     c2 + c3,  -c3],
    [0,      -c3,        c3]
])

def solve_modes(K, M):
    """
    求解廣義特徵值問題，回傳模態頻率(Hz)、自然振動頻率(rad/s)與歸一化模態向量
    """
    eigvals, eigvecs = eigh(K, M)
    omega_n = np.sqrt(eigvals)
    frequencies = omega_n / (2 * np.pi)
    normalized_modes = eigvecs / np.max(np.abs(eigvecs), axis=0)
    return frequencies, omega_n, normalized_modes

# 使用範例
frequencies, omega_n, normalized_modes = solve_modes(K, M)

df_modes = pd.DataFrame(normalized_modes, columns=[f"Mode {i+1}" for i in range(3)])
df_freq = pd.DataFrame({
    "Mode": [f"Mode {i+1}" for i in range(3)],
    "Frequency (Hz)": frequencies,
    "Omega_n (rad/s)": omega_n
})
modal_masses = []
for i in range(3):
    phi_i = normalized_modes[:, i].reshape(-1, 1)
    Mi = float(phi_i.T @ M @ phi_i)
    modal_masses.append(Mi)

print(df_freq)
print(df_modes)
print(f"M_1 = {modal_masses[0]:.6e}")
print(f"M_2 = {modal_masses[1]:.6e}")
print(f"M_3 = {modal_masses[2]:.6e}")
