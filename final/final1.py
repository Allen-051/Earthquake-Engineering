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


# 質量協調阻尼器設定
# 阻尼器七參數
# 質量比
md_r_7 = 0.03
md_7 = md_r_7 * m1  # 阻尼器質量
# 協調頻率比
omega_r_7 = 0.9592
omegad_7 = omega_r_7 * np.sqrt(k1 / m1)  # 阻尼器的頻率
Kd_7 = md_7 * omegad_7 ** 2  # 阻尼器的勁度
# 阻尼比
zeta_r_7 = 0.0857
Cd_7 = zeta_r_7 * (2 * np.sqrt(k1 / m1))  # 阻尼器的阻尼比

# 計算加入阻尼器的運動方程式
def calculate_motion_with_damper(m1, m2, m3, md, k1, k2, k3, kd, c1, c2, c3, cd):
    """
    計算加入阻尼器後的運動方程式矩陣
    """
    # 更新質量矩陣
    M_d = np.diag([m1, m2, m3, md])
    
    # 更新勁度矩陣
    K_d = np.array([
    [k1 + k2, -k2,       0,        0],
    [-k2,     k2 + k3,  -k3,       0],
    [0,      -k3,        k3+kd,  -kd],
    [0,       0,        -kd,      kd]])
    
    # 更新阻尼矩陣
    C_d = np.array([
    [c1 + c2, -c2,       0,        0],
    [-c2,     c2 + c3,  -c3,       0],
    [0,      -c3,        c3+cd,  -cd],
    [0,       0,        -cd,      cd]])

    return K_d, M_d, C_d

# 用線性加速度法(Newmark-beta法)計算位移反應
# 先讀取地震加速度資料
def read_acceleration_data(file_path):
    """
    讀取地震加速度資料
    """
    data = pd.read_csv(file_path, header=None)
    time = data.iloc[:, 0].values
    acceleration = data.iloc[:, 1].values
    # 詢問使用者儲存檔案路徑
    save_path = input("請輸入結果儲存路徑: ").strip()
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    return time, acceleration, save_path

def newmark_beta_method(K, M, C, ag, dt, gamma=0.5, beta=0.25):
    """
    使用Newmark-beta法計算位移、速度和加速度
    """
    N = len(ag)
    dof = M.shape[0]

    # 初始化位移、速度和加速度
    u = np.zeros((N, dof))
    v = np.zeros((N, dof))
    a = np.zeros((N, dof))

    # 初始條件
    a[0] = np.linalg.solve(M, ag[0] * np.ones(dof))

    # 計算系統的等效質量矩陣
    A = M + gamma * dt * C + beta * dt**2 * K

    for i in range(1, N):
        # 計算等效力向量
        F_eq = M @ (ag[i] * np.ones(dof)) + C @ v[i-1] + K @ u[i-1]

        # 更新位移、速度和加速度
        u[i] = np.linalg.solve(A, F_eq)
        a[i] = (u[i] - u[i-1]) / dt - (1 - gamma) * v[i-1]
        v[i] = v[i-1] + (gamma * dt) * a[i]

    return u, v, a

# 繪圖
def plot_results(time, u, v, a, save_path):
    """
    繪製位移、速度和加速度的時間歷程圖
    """

    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(time, u[:, 0], label='Displacement (m)', color='blue')
    plt.title('Displacement Time History')
    plt.xlabel('Time (s)')
    plt.ylabel('Displacement (m)')
    plt.grid(False)
    plt.savefig(os.path.join(save_path, 'u-t.png'), dpi=300)
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(time, v[:, 0], label='Velocity (m/s)', color='orange')
    plt.title('Velocity Time History')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.grid(False)
    plt.savefig(os.path.join(save_path, 'v-t.png'), dpi=300)
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(time, a[:, 0], label='Acceleration (m/s²)', color='red')
    plt.title('Acceleration Time History')
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (m/s²)')
    plt.grid(False)
    plt.savefig(os.path.join(save_path, 'a-t.png'), dpi=300)
    plt.legend()

    plt.tight_layout()
    plt.show()

# 主程式
def main():

    print(df_freq)
    print(df_modes)

    # 輸出結果
    for i, Mi in enumerate(modal_masses, 1):
        print(f"Mode {i} 的正規化模態質量 M{i} = {Mi:.2e} kg")

    # 讀取地震加速度資料
    file_path = input("請輸入地震加速度資料檔案路徑: ").strip()
    time, ag, save_path = read_acceleration_data(file_path)

    # 計算加入阻尼器後的運動方程式
    K_d, M_d, C_d = calculate_motion_with_damper(m1, m2, m3, md_7, k1, k2, k3, Kd_7, c1, c2, c3, Cd_7, omegad_7)

    # 使用Newmark-beta法計算位移、速度和加速度
    dt = time[1] - time[0]  # 假設時間間隔一致
    u, v, a = newmark_beta_method(K_d, M_d, C_d, ag, dt)

    # 繪製結果
    plot_results(time, u, v, a, save_path)