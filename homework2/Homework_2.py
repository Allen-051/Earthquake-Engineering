import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def get_user_input():
    """取得使用者輸入的參數"""
    file_path = input("請輸入地震歷時檔案完整路徑（例如 C:/.../Northridge_NS.txt 或 .csv）：")
    save_path = input("請輸入要儲存結果的資料夾路徑：").strip()
    filename = input("請輸入檔案名稱（不含副檔名）：").strip()
    W_kip = float(input("請輸入結構重量 W (kips)："))
    k_kip_in = float(input("請輸入結構勁度 k (kip/in)："))
    zeta = float(input("請輸入阻尼比 ζ (0~1)："))
    u0 = float(input("請輸入初始位移 u0 (in)："))
    v0 = float(input("請輸入初始速度 v0 (in/s)："))
    dt = float(input("請輸入時間間隔 Δt (s)："))
    gamma = float(input("請輸入 Newmark 參數 δ (例如 0.5)："))  # δ → gamma
    beta = float(input("請輸入 Newmark 參數 α (例如 0.25)："))  # α → beta
    return file_path, save_path, filename, W_kip, k_kip_in, zeta, u0, v0, dt, gamma, beta

def compute_structure_parameters(W_kip, k_kip_in, zeta):
    """計算結構參數"""
    W_lbf = W_kip * 1000
    m = W_lbf / 386.1  # slug
    k = k_kip_in * 1000  # lbf/in
    wn = np.sqrt(k / m)
    c = 2 * zeta * m * wn
    return m, k, c

def read_earthquake_data(file_path):
    """讀取地震歷時資料"""
    if file_path.endswith('.txt'):
        data = np.loadtxt(file_path)
    elif file_path.endswith('.csv'):
        data = pd.read_csv(file_path).values
    else:
        raise ValueError("不支援的檔案格式，請提供 .txt 或 .csv 檔案")
    time = data[:, 0]
    ag = data[:, 1] * 386.1  # g → in/s²
    return time, ag

def perform_time_history_analysis(time, ag, m, k, c, u0, v0, dt, gamma, beta):
    """執行時間歷程分析"""
    npts = len(ag)
    u = np.zeros(npts)
    v = np.zeros(npts)
    a_resp = np.zeros(npts)

    u[0] = u0
    v[0] = v0
    p = -m * ag  # 慣性力
    a_resp[0] = (p[0] - c * v[0] - k * u[0]) / m

    k_eff = k + gamma / (beta * dt) * c + m / (beta * dt ** 2)

    for i in range(1, npts):
        rhs = p[i] + \
              m * ((1 / (beta * dt ** 2)) * u[i - 1] + (1 / (beta * dt)) * v[i - 1] + (1 / (2 * beta) - 1) * a_resp[i - 1]) + \
              c * ((gamma / (beta * dt)) * u[i - 1] + ((gamma / beta) - 1) * v[i - 1] + dt * ((gamma / (2 * beta)) - 1) * a_resp[i - 1])

        u[i] = rhs / k_eff
        a_resp[i] = (1 / (beta * dt ** 2)) * (u[i] - u[i - 1] - dt * v[i - 1]) - (1 - 2 * beta) / (2 * beta) * a_resp[i - 1]
        v[i] = v[i - 1] + dt * ((1 - gamma) * a_resp[i - 1] + gamma * a_resp[i])

    return u, v, a_resp

def save_results_to_csv(time, u, v, a_resp, save_path, filename):
    """儲存結果到 CSV"""
    full_path = os.path.join(save_path, f"{filename}.csv")

    max_u = np.max(np.abs(u))
    max_v = np.max(np.abs(v))
    max_a = np.max(np.abs(a_resp))

    df = pd.DataFrame({
        'Time (s)': time,
        'Displacement (in)': u,
        'Velocity (in/s)': v,
        'Acceleration (in/s²)': a_resp
    })

    max_row = pd.DataFrame({
        'Time (s)': ['MAX'],
        'Displacement (in)': [max_u],
        'Velocity (in/s)': [max_v],
        'Acceleration (in/s²)': [max_a]
    })

    df = pd.concat([df, pd.DataFrame([{}]), max_row], ignore_index=True)
    df.to_csv(full_path, index=False)
    print(f"結果已儲存至：{full_path}")

def plot_results(time, u, v, a_resp, save_path, filename):
    """繪製結果圖表，並儲存到指定路徑"""
    # 建立完整的儲存路徑
    displacement_path = os.path.join(save_path, f"{filename}_displacement.png")
    velocity_path = os.path.join(save_path, f"{filename}_velocity.png")
    acceleration_path = os.path.join(save_path, f"{filename}_acceleration.png")

    # 繪製位移圖
    plt.figure(figsize=(10, 6))
    plt.plot(time, u, color='black', linestyle='-', label='Displacement')
    plt.title("Displacement vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Displacement (in)")
    plt.grid(True)
    plt.savefig(displacement_path)
    plt.show()

    # 繪製速度圖
    plt.figure(figsize=(10, 6))
    plt.plot(time, v, color='red', linestyle='-', label='Velocity')
    plt.title("Velocity vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (in/s)")
    plt.grid(True)
    plt.savefig(velocity_path)
    plt.show()

    # 繪製加速度圖
    plt.figure(figsize=(10, 6))
    plt.plot(time, a_resp, color='blue', linestyle='-', label='Acceleration')
    plt.title("Acceleration vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (in/s²)")
    plt.grid(True)
    plt.savefig(acceleration_path)
    plt.show()

def main():
    """主程式流程"""
    file_path, save_path, filename, W_kip, k_kip_in, zeta, u0, v0, dt, gamma, beta = get_user_input()
    m, k, c = compute_structure_parameters(W_kip, k_kip_in, zeta)
    time, ag = read_earthquake_data(file_path)
    u, v, a_resp = perform_time_history_analysis(time, ag, m, k, c, u0, v0, dt, gamma, beta)
    
    # 儲存結果到 CSV
    save_results_to_csv(time, u, v, a_resp, save_path, filename)
    
    # 繪製圖表並儲存圖片
    plot_results(time, u, v, a_resp, save_path, filename)

if __name__ == "__main__":
    main()