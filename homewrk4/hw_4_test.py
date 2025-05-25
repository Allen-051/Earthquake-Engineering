#初始位移 u0 = 0 ; 
#初始速度 v0 = 40 (inch/s); 
#降伏位移為 x_y = 1 (inch);
#單自由度結構物質量 m = 1 (kip*s^2/inch);
#結構物的初始勁度 k1 = 631.65 (kip/inch);
#結構物的降伏勁度 k2 = 126.33 (kip/inch);
#結構物的阻尼比 zeta = 0.05;
#時間間隔 dt = 0.005 (s);
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import os
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']  # 設定中文字型
plt.rcParams['axes.unicode_minus'] = False  # 正確顯示負號

def get_user_input():
    """取得使用者輸入的參數"""
    
    save_path = input("請輸入要儲存結果的資料夾路徑（預設為桌面）：").strip()
    if not save_path:
        save_path = os.path.join(os.path.expanduser("~"), "Desktop")
        print(f"未輸入路徑，自動儲存至：{save_path}")
    filename = input("請輸入檔案名稱（不含副檔名）：").strip()
    m = float(input("請輸入結構質量 M (kip*s^2/inch)："))
    k1 = float(input("請輸入結構勁度 k1 (kip/in)："))
    k2 = float(input("請輸入結構勁度 k2 (kip/in)："))
    zeta = float(input("請輸入阻尼比 ζ (0~1)："))
    u0 = float(input("請輸入初始位移 u0 (in)："))
    v0 = float(input("請輸入初始速度 v0 (in/s)："))
    dt = float(input("請輸入時間間隔 Δt (s)："))
    F0 = float(input("請輸入結構物的外力 F0 (kip)："))

    return  save_path, filename, m, k1, k2, zeta, u0, v0, dt, F0

def matrix(m, v0, u0, k1, k2, zeta, dt, F0):
    """計算參數"""
    # average acceleration method
    delta = 0.5
    alpha = 0.25
    c1 = 2 * zeta * math.sqrt(m * k1)  # 阻尼係數
    c2 = 2 * zeta * math.sqrt(m * k2)  # 阻尼係數
    beta_1 = m + c1 * delta * dt + k1 * alpha * dt**2
    beta_2 = m + c2 * delta * dt + k2 * alpha * dt**2

    # 起始加速度
    a0_1 = -c1/m * v0 - k1/m * u0 + F0 / m
    a0_2 = -c2/m * v0 - k2/m * u0 + F0 / m
    # 定義A_1矩陣
    a11 = 1
    a12 = dt - k1 * alpha * dt**3/beta_1
    a13 = 0.5 * dt**2 - c1 * alpha *dt**3 / beta_1 - 0.5 * k1 * alpha * dt**4 / beta_1
    a21 = 0
    a22 = 1 - k1 * alpha * dt**2 / beta_1
    a23 = dt - c1 * alpha * dt**2 / beta_1 - k1 * alpha * dt**3 / (2 * beta_1)
    a31 = 0
    a32 = -k1 * dt / beta_1
    a33 = 1 - c1 * dt / beta_1 - k1 * dt**2 / (2 * beta_1)

    A_1 = np.array([[a11, a12, a13 ], [a21, a22, a23] ,[a31, a32, a33]])

    # 定義B_1向量
    b11 = alpha * dt**2 / beta_1
    b21 = delta * dt / beta_1
    b31 = 1 / beta_1

    B_1 = np.array([[b11], [b21], [b31]])

    # 定義A_2矩陣
    a11 = 1
    a12 = dt - k2 * alpha * dt**3/beta_2
    a13 = 0.5 * dt**2 - c2 * alpha *dt**3 / beta_2 - 0.5 * k2 * alpha * dt**4 / beta_2
    a21 = 0
    a22 = 1 - k2 * alpha * dt**2 / beta_2
    a23 = dt - c2 * alpha * dt**2 / beta_2 - k2 * alpha * dt**3 / (2 * beta_2)
    a31 = 0
    a32 = -k2 * dt / beta_2
    a33 = 1 - c2 * dt / beta_2 - k2 * dt**2 / (2 * beta_2)

    A_2 = np.array([[a11, a12, a13 ], [a21, a22, a23] ,[a31, a32, a33]])

    # 定義B_2向量
    b11 = alpha * dt**2 / beta_2
    b21 = delta * dt / beta_2
    b31 = 1 / beta_2

    B_2 = np.array([[b11], [b21], [b31]])

    print("A_1矩陣：\n", A_1)
    print("B_1向量：\n", B_1)
    print("A_2矩陣：\n", A_2)
    print("B_2向量：\n", B_2)

    return A_1, B_1, A_2, B_2, a0_1, a0_2

def calculate_displacement(A_1, B_1, A_2, B_2, m, k1, k2, zeta, u0, v0, dt, a0_1 ,a0_2):
    """計算位移"""
    # x_j+1 = A_1 * x_j + B_1 * (F_j+1 - F_j)
    

def test_displacement_loop(A_1, B_1, dt, u0, v0, a0_1, F):
    """
    測試 x_j+1 = A_1 * x_j + B_1 * (F_j+1 - F_j) 的迴圈
    初始條件: x_0 = [[u0], [v0], [a0_1]]
    F: 外力陣列，長度需 >= 17
    """
    steps = 16
    x_hist = []
    x_j = np.array([[u0], [v0], [a0_1]])
    x_hist.append(x_j.copy())

    for j in range(steps):
        delta_F = F[j+1] - F[j]
        x_j1 = np.matmul(A_1, x_j) + B_1 * delta_F
        x_hist.append(x_j1.copy())
        x_j = x_j1

    # 將結果轉為 numpy array 方便檢查
    x_hist = np.hstack(x_hist)  # shape: (3, steps+1)
    # 新增時間列
    time = np.arange(x_hist.shape[1]) * dt
    x_time = np.vstack([time, x_hist])  # 第一行為時間

    df = pd.DataFrame(x_time.T, columns=['time', 'u', 'v', 'a'])
    print(df.to_string(index=False))
    return x_time

def plot_displacement(x_hist, save_path, filename):
    """繪製位移圖"""
    plt.figure(figsize=(10, 5))
    plt.plot(x_hist[1], label='位移 u (in)', marker='o')
    plt.plot(x_hist[2], label='速度 v (in/s)', marker='x')
    plt.plot(x_hist[3], label='加速度 a (in/s^2)', marker='s')
    plt.title('位移、速度和加速度隨時間的變化')
    plt.xlabel('時間步數')
    plt.ylabel('數值')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_path, f"{filename}.jpg"))
    plt.show()
def main():
    save_path, filename, m, k1, k2, zeta, u0, v0, dt, F0 = get_user_input()
    
    # 計算矩陣和向量
    A_1, B_1, A_2, B_2, a0_1, a0_2 = matrix(m, v0, u0, k1, k2, zeta, dt, F0)
    
    # 假設外力 F 為一個長度為 17 的陣列，這裡僅作為測試用
    F = np.zeros(17)
    F[0] = F0  # 初始外力
    for i in range(1, 17):
        F[i] = F[i-1]   # 模擬外力逐漸增加

    # 測試位移迴圈
    x_hist = test_displacement_loop(A_1, B_1, dt, u0, v0, a0_1, F)
    # 將結果存入 DataFrame 並儲存
    df = pd.DataFrame(x_hist.T, columns=['time', 'u', 'v', 'a'])
    df.to_csv(f"{save_path}/{filename}.csv", index=False)
    print(f"結果已儲存至 {save_path}/{filename}.csv")
    # 繪製位移圖
    plot_displacement(x_hist,save_path, filename)
    
    return x_hist

if __name__ == "__main__":
    x_hist = main()
    