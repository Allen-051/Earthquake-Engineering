import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# Step 0: 詢問使用者單位系統
while True:
    unit_system = input("Please select the unit system:\n1 for Imperial units\n2 for Metric units\nYour choice: ")
    if unit_system == "1":
        print("You have selected Imperial units.")
        break
    elif unit_system == "2":
        print("You have selected Metric units.")
        break
    else:
        print("Invalid input. Please enter 1 for Imperial units or 2 for Metric units.")

# Step 1: 讀取檔案並提取欄位
while True:
    file_type = input("Please enter the file type:\n1 for CSV file\n2 for TXT file\nYour choice: ")
    file_path = input("Enter the path to the file: ").strip().strip('"')
    file_path = file_path.replace("\\", "\\\\")
    try:
        if file_type == "1":
            # CSV 檔案處理
            print("Processing CSV file...")
            data = np.genfromtxt(file_path, delimiter=",", skip_header=1, dtype=float)
            time_column = data[:, 0]  # 第一欄為時間
            P_column = data[:, 1]    # 第二欄為加速度或外力
            print("CSV file loaded successfully.")
        elif file_type == "2":
            # TXT 檔案處理
            print("Processing TXT file...")
            with open(file_path, 'r') as f:
                headers = f.readline().strip().split()  # 讀取第一行並分割成關鍵字
                print("偵測並讀取關鍵字:", headers)

                # 確定 "Time(s)" 和 "Acc(g)" 的索引
                time_index = headers.index("Time(s)")
                acc_index = headers.index("Acc(g)")

            # 使用 np.genfromtxt 讀取數據，略過標題行
            data = np.genfromtxt(file_path, skip_header=1)
            time_column = data[:, time_index]
            P_column = data[:, acc_index]
            print("TXT file loaded successfully.")
            # 偵測並提取 "time" 和 "value" 對應的欄位
            print("time：")
            print(", ".join(map(str, time_column)))
            print("P or ground movement：")
            print(", ".join(map(str, P_column)))
        else:
            print("Invalid input. Please enter 1 for CSV or 2 for TXT.")
            continue
        break
    except Exception as e:
        print(f"Error reading file: {e}. Please try again.")



# Step 2: 檢查時間間距
delta_t = np.diff(time_column)
consistent = np.allclose(delta_t, delta_t[0])

for i in range(len(delta_t)):
    if i < len(delta_t) - 1 and delta_t[i] != delta_t[i + 1]:
        print("False")
        break
else:
    print("True")

# 如果時間間距不一致，進行內插修正
if not consistent:
    dt = np.min(delta_t)
    new_time = np.arange(time_column[0], time_column[-1] + dt, dt)
    new_values = np.interp(new_time, time_column, P_column)
    time_column, P_column = new_time, new_values
    print("Time intervals have been interpolated.")

# 定義函數以接受公制單位參數
def ask_SI_parameters():
    m = float(input("Please enter the mass (unit: kg) of the system: "))
    zeta = float(input("Please enter the damping ratio (zeta) of the system: "))
    k = float(input("Please enter the stiffness (unit: N/m) of the system: "))
    force = float(input("Please enter the amplitude of force (unit: N) applied on the structure: "))
    omega = float(input("Please enter the frequency of external force (rad/s): "))
    u0 = float(input("Please enter the initial displacement (unit: m) of the system: "))
    v0 = float(input("Please enter the initial moving velocity (unit: m/s) of the system: "))
    Par = int(input("Enter 1 if the second column of the input data is external forces.\nEnter 2 if the second column of the input data is ground movement acceleration.\nPlease enter 1 or 2: "))
    return m, zeta, k, force, omega, u0, v0, Par

# 定義函數以接受英制單位參數
def ask_imperial_parameters():
    m = float(input("Please enter the mass (unit: kip) of the system: ")) * 14.5939  # kip to kg
    zeta = float(input("Please enter the damping ratio (zeta) of the system: "))
    k = float(input("Please enter the stiffness (unit: kip/in) of the system: ")) * 1129.848  # ksi/in to N/m
    force = float(input("Please enter the amplitude of force (unit: kip) applied on the structure: ")) * 4448.22  # kip to N
    omega = float(input("Please enter the frequency of external force (rad/s): "))
    u0 = float(input("Please enter the initial displacement (unit: in) of the system: ")) * 0.0254  # in to m
    v0 = float(input("Please enter the initial moving velocity (unit: in/s) of the system: ")) * 0.0254  # in/s to m/s
    Par = int(input("Enter 1 if the second column of the input data is external forces.\nEnter 2 if the second column of the input data is ground movement acceleration.\nPlease enter 1 or 2: "))
    return m, zeta, k, force, omega, u0, v0, Par

# 根據單位系統選擇對應的函數
if unit_system == "1":
    print("Using Imperial units.")
    m, zeta, k, force, omega, u0, v0, Par = ask_imperial_parameters()
elif unit_system == "2":
    print("Using Metric units.")
    m, zeta, k, force, omega, u0, v0, Par = ask_SI_parameters()

# 計算阻尼係數
c = 2 * m * zeta * np.sqrt(k / m)

# Step 4: 定義計算方法
if Par == 2:
    P_column = -m * P_column

def harmonic_exact(zeta, omega, k, force_amplitude, time, u0, v0):
    # 外力參數
    F0 = force_amplitude

    omega_n = omega_n = np.sqrt(k / m)
    # 計算系統參數
    omega_d = omega_n * np.sqrt(1 - zeta ** 2)  # 阻尼振動頻率 (rad/s)
    beta = omega / omega_n
    denominator = (1 - beta ** 2) ** 2 + (2 * zeta * beta) ** 2

    # 穩態解 (特解)
    A_p = (F0 / k) / np.sqrt(denominator)
    phi = np.arctan2(2 * zeta * beta, 1 - beta ** 2)
    u_p = lambda t: A_p * np.sin(omega * t - phi)

    # 修正瞬態解常數
    C1 = u0 - u_p(0)
    C2 = (v0 + zeta * omega_n * C1 - omega * A_p * np.cos(phi)) / omega_d

    # 瞬態解 (齊次解)
    u_h = lambda t: np.exp(-zeta * omega_n * t) * (C1 * np.cos(omega_d * t) + C2 * np.sin(omega_d * t))

    # 總解
    u = u_h(time) + u_p(time)
    return u
def central_difference_method(m, c, k, p, time, u0, v0):
    dt = time[1] - time[0]
    n = len(time)
    u = np.zeros(n)
    v = np.zeros(n)
    a = np.zeros(n)
    u[0] = u0
    v[0] = v0
    a[0] = (p[0] - c * v[0] - k * u[0]) / m
    k_eff = m / dt**2 + c / (2 * dt)
    u[-1] = u[0] - dt * v[0] + 0.5 * a[0] * dt ** 2
    A = m / dt**2 - c/ (2 * dt)
    B = k - 2 * m / dt**2
    for i in range(0, n-1):
        u[i+1] = (p[i] - A * u[i-1] - B * u[i]) / k_eff
        v[i] = (u[i+1] - u[i-1]) / (2 * dt)
        a[i] = (u[i+1] - 2 * u[i] + u[i-1]) / dt**2
    return u, v, a

def linear_acceleration_method(m, c, k, p, time, u0, v0):
    dt = time[1] - time[0]  # 時間步長
    n = len(time)  # 時間步數
    # 初始化位移、速度、加速度陣列
    u = np.zeros(n)
    v = np.zeros(n)
    a = np.zeros(n)

    # 初始條件
    u[0] = u0
    v[0] = v0
    a[0] = (p[0] - c * v[0] - k * u[0]) / m  # 計算初始加速度

    # Newmark 線性加速度法的參數
    gamma = 0.5  # Newmark 方法中的速度加權因子
    beta = 1 / 6  # Newmark 方法中的位移加權因子

    # 預計算常數
    k_eff = k + (gamma * c) / (beta * dt) + m / (beta * dt**2)
    A = m / (beta * dt) + c *( gamma / beta)
    B = m / (2 * beta) + dt *( gamma /(2 * beta) -1)

    # Newmark 迭代
    for i in range(0 , n-1):
        delta_p_head = (p[i+1] - p[i]) + A * v[i] + B * a[i]
        delta_u = delta_p_head/ k_eff
        delta_v = gamma* delta_u/(beta* dt) - gamma * v[i]/beta + dt * a[i] * (1 - gamma /(2 * beta))
        delta_a = delta_u/(beta* dt**2) - v[i]/(beta*dt) - a[i]/(2 *beta)
        u[i+1] = u[i] + delta_u
        v[i+1] = v[i] + delta_v
        a[i+1] = a[i] + delta_a

    return u, v, a
u_h = harmonic_exact(zeta, omega, k, force, time_column, u0, v0)
u1, v1, a1 = central_difference_method(m, c, k, P_column, time_column, u0, v0)
u2, v2, a2 = linear_acceleration_method(m, c, k, P_column, time_column, u0, v0)

plt.figure(figsize=(10, 6))
plt.plot(time_column, u_h, label="u_Exact", color="black",linestyle="-.")
plt.plot(time_column, u1, label="u1-C.D.M", color="green",linestyle="-")
plt.plot(time_column, u2, label="u2-N.L.A.M", color="purple", linestyle="-")
plt.xlabel("Time (s)")
plt.ylabel("Displacement (u)")
plt.title("Displacement Response of SDOF System")
plt.xlim(0, 6)
plt.ylim(-3, 3)
plt.legend()
plt.grid(True)


# Step 5: 輸出到檔案
header = ['time(t)', 'H.Exact(u)', 'C.D(u1)', 'C.D(v1)', 'C.D(a1)' ,'N.L.A(u2)', 'N.L.A(v2)', 'N.L.A(a2)' ]
response = np.column_stack((time_column, u_h, u1, v1, a1, u2, v2, a2))
result = [header] + response.tolist()
if Par == 1:
    print('Dear user, the input data is an external force form.\n')
    for data in result:
        print(",   ".join(map(str, data)))
if Par == 2:
    print('Dear user, the input data is a ground acceleration movement.\n')
    for data in result:
        print(",   ".join(map(str, data)) )
output_path = r"D:\\Earthquake-Engineering\\homework2\\Northridge_result.txt"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "w") as f:
    for row in result:
        f.write(", ".join(map(str, row)) + "\n")

print(f"Response data saved to: {output_path}")

save_path = input("Enter the save path for the image: ")
plt.savefig(save_path)
print(f"Plot saved to {save_path}")

plt.show()