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

    return file_path, save_path, filename, W_kip, k_kip_in, zeta, u0, v0, dt

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

def average_acceleration_method(time, ag, m, k, c, u0, v0, dt):
    """使用平均加速度法計算時間歷程分析"""
    gamma = 0.5
    beta = 0.25
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

def linear_acceleration_method(time, ag, m, k, c, u0, v0, dt):
    """使用線性加速度法計算時間歷程分析"""
    gamma = 0.5
    beta = 1 / 6
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

def central_difference_method(time, ag, m, k, c, u0, v0, dt):
    """使用中央差分法計算時間歷程分析"""
    npts = len(ag)
    u = np.zeros(npts)
    v = np.zeros(npts)
    a_resp = np.zeros(npts)

    # 初始條件
    u[0] = u0
    v[0] = v0
    a_resp[0] = (1 / m) * (-c * v[0] - k * u[0] - m * ag[0])

    # 預計算常數
    k_eff = m / dt**2 + c / (2 * dt)
    a = m / dt**2 - c / (2 * dt)
    b = k - 2 * m / dt**2

    # 時間歷程計算
    for i in range(1, npts - 1):
        p_eff = -m * ag[i] - a * u[i - 1] - b * u[i]
        u[i + 1] = p_eff / k_eff
        v[i] = (u[i + 1] - u[i - 1]) / (2 * dt)
        a_resp[i] = (u[i + 1] - 2 * u[i] + u[i - 1]) / dt**2

    # 最後一點速度和加速度
    v[-1] = (u[-1] - u[-2]) / dt
    a_resp[-1] = (u[-1] - 2 * u[-2] + u[-3]) / dt**2

    return u, v, a_resp

def wilsons_theta_method(time, ag, m, k, c, u0, v0, dt, theta=1.4, tol=1e-6, max_iter=100):
    """改進的 Wilson's θ 方法計算時間歷程分析"""
    gamma = 0.5
    beta = 1 / 6
    npts = len(ag)
    u = np.zeros(npts)
    v = np.zeros(npts)
    a_resp = np.zeros(npts)

    # 初始條件
    u[0] = u0
    v[0] = v0
    a_resp[0] = (1 / m) * (-c * v[0] - k * u[0] - m * ag[0])

    # 預計算常數
    k_eff = k + (theta / (beta * dt ** 2)) * m + (theta * gamma / (beta * dt)) * c

    for i in range(1, npts):
        # 預測力項
        p_theta = -m * ag[i] + (1 - theta) * (-m * ag[i - 1]) + \
                  m * ((1 - theta) * (1 / (beta * dt ** 2)) * u[i - 1] + (1 - theta) * (1 / (beta * dt)) * v[i - 1]) + \
                  c * ((1 - theta) * (gamma / (beta * dt)) * u[i - 1] + (1 - theta) * ((gamma / beta) - 1) * v[i - 1])

        # 初始位移估計
        u_theta = u[i - 1]
        for _ in range(max_iter):
            # 計算內力
            
            f_int = k * u_theta + c * ((gamma / (beta * dt)) * (u_theta - u[i - 1]) + (1 - gamma / beta) * v[i - 1] + dt * (1 - gamma / (2 * beta)) * a_resp[i - 1])
            # 更新位移
            delta_u = (p_theta - f_int) / k_eff
            u_theta += delta_u
            # 檢查收斂
            if np.abs(delta_u).max() < tol:
                break

        # 更新位移
        u[i] = u_theta

        # 更新加速度
        a_resp[i] = (1 / (beta * dt ** 2)) * (u[i] - u[i - 1] - dt * v[i - 1]) - (1 - 2 * beta) / (2 * beta) * a_resp[i - 1]

        # 更新速度
        v[i] = v[i - 1] + dt * ((1 - gamma) * a_resp[i - 1] + gamma * a_resp[i])

    return u, v, a_resp

def save_results_to_csv(time, u1, v1, a1, u2, v2, a2, u3, v3, a3, u4, v4, a4, save_path, filename):
    """儲存所有方法的結果到 CSV"""
    full_path = os.path.join(save_path, f"{filename}.csv")

    # 計算每種方法的最大值
    max_values = {
        "Average Acceleration Method": {
            "Max Displacement (in)": np.max(np.abs(u1)),
            "Max Velocity (in/s)": np.max(np.abs(v1)),
            "Max Acceleration (in/s²)": np.max(np.abs(a1))
        },
        "Linear Acceleration Method": {
            "Max Displacement (in)": np.max(np.abs(u2)),
            "Max Velocity (in/s)": np.max(np.abs(v2)),
            "Max Acceleration (in/s²)": np.max(np.abs(a2))
        },
        "Central Difference Method": {
            "Max Displacement (in)": np.max(np.abs(u3)),
            "Max Velocity (in/s)": np.max(np.abs(v3)),
            "Max Acceleration (in/s²)": np.max(np.abs(a3))
        },
        "Wilson's Method": {
            "Max Displacement (in)": np.max(np.abs(u4)),
            "Max Velocity (in/s)": np.max(np.abs(v4)),
            "Max Acceleration (in/s²)": np.max(np.abs(a4))
        }
    }

    # 建立 DataFrame
    df = pd.DataFrame({
        "Time (s)": time,
        "Avg Acc Displacement (in)": u1,
        "Avg Acc Velocity (in/s)": v1,
        "Avg Acc Acceleration (in/s²)": a1,
        "Lin Acc Displacement (in)": u2,
        "Lin Acc Velocity (in/s)": v2,
        "Lin Acc Acceleration (in/s²)": a2,
        "Central Diff Displacement (in)": u3,
        "Central Diff Velocity (in/s)": v3,
        "Central Diff Acceleration (in/s²)": a3,
        "Wilson's Displacement (in)": u4,
        "Wilson's Velocity (in/s)": v4,
        "Wilson's Acceleration (in/s²)": a4
    })

    # 空出一列，然後附加最大值
    df = pd.concat([df, pd.DataFrame([{}])], ignore_index=True)  # 空白列
    for method, values in max_values.items():
        max_row = pd.DataFrame({
            "Time (s)": [f"MAX ({method})"],
            "Avg Acc Displacement (in)": [values.get("Max Displacement (in)", "")],
            "Avg Acc Velocity (in/s)": [values.get("Max Velocity (in/s)", "")],
            "Avg Acc Acceleration (in/s²)": [values.get("Max Acceleration (in/s²)", "")]
        })
        df = pd.concat([df, max_row], ignore_index=True)

    # 儲存到 CSV
    df.to_csv(full_path, index=False)
    print(f"結果已儲存至：{full_path}")

def plot_results(time, u1, v1, a1, u2, v2, a2, u3, v3, a3, u4, v4, a4, save_path, filename):
    """繪製結果圖表，並儲存到指定路徑"""
    # 建立完整的儲存路徑
    displacement_path = os.path.join(save_path, f"{filename}_displacement.png")
    velocity_path = os.path.join(save_path, f"{filename}_velocity.png")
    acceleration_path = os.path.join(save_path, f"{filename}_acceleration.png")

    # 繪製位移圖
    plt.figure(figsize=(10, 6))
    plt.plot(time, u1, color='black', linestyle='-', linewidth=1.5, label='Average Acceleration Method')
    plt.plot(time, u2, color='blue', linestyle='--', linewidth=1.5, label='Linear Acceleration Method')
    plt.plot(time, u3, color='red', linestyle='-.', linewidth=1.5, label='Central Difference Method')
    plt.plot(time, u4, color='green', linestyle=':', linewidth=1.5, label="Wilson's Method")
    plt.title("Displacement vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Displacement (in)")
    plt.legend()
    plt.grid(True)
    plt.savefig(displacement_path)
    plt.show()

    # 繪製速度圖
    plt.figure(figsize=(10, 6))
    plt.plot(time, v1, color='black', linestyle='-', linewidth=1.5, label='Average Acceleration Method')
    plt.plot(time, v2, color='blue', linestyle='--', linewidth=1.5, label='Linear Acceleration Method')
    plt.plot(time, v3, color='red', linestyle='-.', linewidth=1.5, label='Central Difference Method')
    plt.plot(time, v4, color='green', linestyle=':', linewidth=1.5, label="Wilson's Method")
    plt.title("Velocity vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (in/s)")
    plt.legend()
    plt.grid(True)
    plt.savefig(velocity_path)
    plt.show()

    # 繪製加速度圖
    plt.figure(figsize=(10, 6))
    plt.plot(time, a1, color='black', linestyle='-', linewidth=1.5, label='Average Acceleration Method')
    plt.plot(time, a2, color='blue', linestyle='--', linewidth=1.5, label='Linear Acceleration Method')
    plt.plot(time, a3, color='red', linestyle='-.', linewidth=1.5, label='Central Difference Method')
    plt.plot(time, a4, color='green', linestyle=':', linewidth=1.5, label="Wilson's Method")
    plt.title("Acceleration vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (in/s²)")
    plt.legend()
    plt.grid(True)
    plt.savefig(acceleration_path)
    plt.show()

def main():
    """主程式流程"""
    file_path, save_path, filename, W_kip, k_kip_in, zeta, u0, v0, dt = get_user_input()
    m, k, c = compute_structure_parameters(W_kip, k_kip_in, zeta)
    time, ag = read_earthquake_data(file_path)

    # 使用平均加速度法計算
    u1, v1, a1 = average_acceleration_method(time, ag, m, k, c, u0, v0, dt)

    # 使用線性加速度法計算
    u2, v2, a2 = linear_acceleration_method(time, ag, m, k, c, u0, v0, dt)

    # 使用中央差分法計算
    u3, v3, a3 = central_difference_method(time, ag, m, k, c, u0, v0, dt)

    # 使用 Wilson's 方法計算
    u4, v4, a4 = wilsons_theta_method(time, ag, m, k, c, u0, v0, dt)

    # 儲存結果到 CSV（僅儲存平均加速度法的結果）
    save_results_to_csv(time, u1, v1, a1, u2, v2, a2, u3, v3, a3, u4, v4, a4, save_path, filename)

    # 繪製圖表並儲存圖片
    plot_results(time, u1, v1, a1, u2, v2, a2, u3, v3, a3, u4, v4, a4, save_path, filename)

if __name__ == "__main__":
    main()