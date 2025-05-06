import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def read_force_file():
    import_path = input("請輸入地震或外力資料檔案路徑（支援txt或csv）：").strip()
    if import_path.endswith('.csv'):
        data = pd.read_csv(import_path, header=None)
    elif import_path.endswith('.txt'):
        data = pd.read_csv(import_path, delim_whitespace=True, header=None)
    else:
        raise ValueError("請提供副檔名為.csv或.txt的檔案")
    
    time_column = data.iloc[:, 0].values
    p_column = data.iloc[:, 1].values
    print("Time Column:", time_column)
    print("P Column:", p_column)
    return time_column, p_column

def central_difference_method(time, m, k, zeta, u0, v0, p):
    dt = time[1] - time[0]
    n = len(time)
    u = np.zeros(n)
    v = np.zeros(n)
    a = np.zeros(n)
    c = 2 * zeta * np.sqrt(m * k)
    u[0] = u0
    v[0] = v0
    a[0] = (p[0] - c * v[0] - k * u[0]) / m
    u[-1] = u[0] - dt * v[0] + 0.5 * a[0] * dt ** 2
    k_eff = m / dt**2 + c / (2 * dt)
    A = m / dt**2 - c / (2 * dt)
    B = k - 2 * m / dt**2
    for i in range(0, n-1):
        u[i+1] = (p[i] - A * u[i-1] - B * u[i]) / k_eff
        v[i] = (u[i+1] - u[i-1]) / (2 * dt)
        a[i] = (u[i+1] - 2 * u[i] + u[i-1]) / dt**2
    return u, v, a

def linear_acceleration_method(time, m, k, zeta, u0, v0, p):
    dt = time[1] - time[0]
    n = len(time)
    u = np.zeros(n)
    v = np.zeros(n)
    a = np.zeros(n)
    c = 2 * zeta * np.sqrt(m * k)
    u[0] = u0
    v[0] = v0
    a[0] = (p[0] - c * v[0] - k * u[0]) / m
    gamma = 0.5
    beta = 1 / 6
    k_eff = k + (gamma * c) / (beta * dt) + m / (beta * dt**2)
    A = m / (beta * dt) + c * (gamma / beta)
    B = m / (2 * beta) + dt * (gamma / (2 * beta) - 1)
    for i in range(n - 1):
        delta_p_head = (p[i+1] - p[i]) + A * v[i] + B * a[i]
        delta_u = delta_p_head / k_eff
        delta_v = gamma * delta_u / (beta * dt) - gamma * v[i] / beta + dt * a[i] * (1 - gamma / (2 * beta))
        delta_a = delta_u / (beta * dt**2) - v[i] / (beta * dt) - a[i] / (2 * beta)
        u[i+1] = u[i] + delta_u
        v[i+1] = v[i] + delta_v
        a[i+1] = a[i] + delta_a
    return u, v, a
import numpy as np

def average_acceleration_method(time, m, k, zeta, u0, v0, p):
    dt = time[1] - time[0]
    n = len(time)

    # 初始化變數
    u = np.zeros(n)
    v = np.zeros(n)
    a = np.zeros(n)

    # 初始條件
    u[0] = u0
    v[0] = v0
    c = 2 * zeta * np.sqrt(m * k)
    a[0] = (p[0] - c * v[0] - k * u[0]) / m

    # Newmark 平均加速度法參數
    beta = 1 / 4
    gamma = 1 / 2

    # 預先計算常數
    k_eff = k + gamma * c / (beta * dt) + m / (beta * dt ** 2)
    a1 = m / (beta * dt) + gamma * c / beta
    a2 = m / (2 * beta) + dt * (gamma / (2 * beta) - 1) * c

    # 時間迭代
    for i in range(n - 1):
        dp = p[i + 1] - p[i] + a1 * v[i] + a2 * a[i]
        du = dp / k_eff
        u[i + 1] = u[i] + du
        v[i + 1] = v[i] + dt * (1 - gamma) * a[i] + gamma * dt * ((du / (beta * dt ** 2)) - v[i] / (beta * dt) - a[i] / (2 * beta))
        a[i + 1] = (du / (beta * dt ** 2)) - v[i] / (beta * dt) - a[i] / (2 * beta)

    return u, v, a

def get_user_parameters():
    while True:
        unit_type = input("請選擇單位系統：\n1. 公制\n2. 英制\n輸入數字：")
        if unit_type == '1':
            m = float(input('結構物質量大小(kg)為何：'))
            k = float(input('結構物勁度(kN/m)為何：')) * 1e3
            zeta = float(input('結構物阻尼比為何：'))
            u0 = float(input('結構物初始位移(m)為何：'))
            v0 = float(input('結構物初始速度(m/s)為何：'))
            return m, k, zeta, u0, v0, 1
        elif unit_type == '2':
            m = float(input('結構物重量大小(k)為何：')) *1000 / 386.4
            k = float(input('結構物勁度(kip/inch)為何：')) * 175.12677
            zeta = float(input('結構物阻尼比為何：'))
            u0 = float(input('結構物初始位移(inch)為何：')) * 0.0254
            v0 = float(input('結構物初始速度(inch/s)為何：')) * 0.0254
            return m, k, zeta, u0, v0, 2
        else:
            print("請輸入正確選項：1 或 2")

def find_maximum(u, v, a, unit_type):
    if unit_type == 2:
        u *= 39.3701
        v *= 39.3701
        a *= 39.3701
        disp_unit = "Displacement(inch)"
        vel_unit = "Velocity(inch/s)"
        acc_unit = "Acceleration(inch/s^2)"
    else:
        disp_unit = "Displacement(m)"
        vel_unit = "Velocity(m/s)"
        acc_unit = "Acceleration(m/s^2)"
    
    max_disp = np.max(np.abs(u))
    max_vel = np.max(np.abs(v))
    max_acc = np.max(np.abs(a))
    
    return {
        disp_unit: max_disp,
        vel_unit: max_vel,
        acc_unit: max_acc
    }

def plot_results(time, u, v, a, unit_type):
    if unit_type == 2:
        u *= 39.3701
        v *= 39.3701
        a *= 39.3701
        disp_unit = "Displacement (inch)"
        vel_unit = "Velocity (inch/s)"
        acc_unit = "Acceleration (inch/s²)"
    else:
        disp_unit = "Displacement (m)"
        vel_unit = "Velocity (m/s)"
        acc_unit = "Acceleration (m/s²)"

    plt.figure(figsize=(10, 6))
    plt.plot(time, u, color='red', linestyle='-', label='Displacement')
    plt.title("Displacement vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel(disp_unit)
    plt.grid(False)
    plt.savefig("displacement.png")
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(time, v, color='blue', linestyle='-', label='Velocity')
    plt.title("Velocity vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel(vel_unit)
    plt.grid(False)
    plt.savefig("velocity.png")
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(time, a, color='purple', linestyle='-', label='Acceleration')
    plt.title("Acceleration vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel(acc_unit)
    plt.grid(False)
    plt.savefig("acceleration.png")
    plt.show()

def save_results_to_csv(time, u, v, a, unit_type):
    if unit_type == 2:
        u *= 39.3701
        v *= 39.3701
        a *= 39.3701
        disp_unit = "Displacement(inch)"
        vel_unit = "Velocity(inch/s)"
        acc_unit = "Acceleration(inch/s^2)"
    else:
        disp_unit = "Displacement(m)"
        vel_unit = "Velocity(m/s)"
        acc_unit = "Acceleration(m/s^2)"

    save_path = input("請輸入要儲存結果的資料夾路徑：").strip()
    filename = input("請輸入檔案名稱（不含副檔名）：").strip()
    full_path = os.path.join(save_path, f"{filename}.csv")
    
    # 計算最大值
    max_values = find_maximum(u, v, a, unit_type)
    
    # 建立 DataFrame
    df = pd.DataFrame({
        "Time(s)": time,
        disp_unit: u,
        vel_unit: v,
        acc_unit: a
    })
    
    # 空出一列，然後附加最大值
    df.loc[len(df)] = [""] * len(df.columns)  # 空白列
    df.loc[len(df)] = ["Maximum"] + list(max_values.values())
    
    # 儲存到 CSV
    df.to_csv(full_path, index=False)
    print(f"結果已儲存至：{full_path}")

# 主程式流程
if __name__ == "__main__":
    time_column, p_column = read_force_file()
    m, k, zeta, u0, v0, unit_type = get_user_parameters()
    method = input("請選擇分析法：1. 中央差分法 2. 線性加速度法 3. 平均加速度法\n輸入數字：")
    if method == '1':
        u, v, a = central_difference_method(time_column, m, k, zeta, u0, v0, p_column)
    elif method == '2':
        u, v, a = linear_acceleration_method(time_column, m, k, zeta, u0, v0, p_column)
    elif method == '3':
        u, v, a = average_acceleration_method(time_column, m, k, zeta, u0, v0, p_column)
    else:
        raise ValueError("請輸入有效選項（1 或 2）")

    plot_results(time_column, u.copy(), v.copy(), a.copy(), unit_type)

    save_option = input("是否要儲存結果為CSV檔？是請輸入 Y/y，否請輸入 N/n：")
    if save_option.lower() == 'y':
        save_results_to_csv(time_column, u.copy(), v.copy(), a.copy(), unit_type)