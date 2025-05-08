import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


# === 使用者輸入 ===
file_path = input("請輸入地震歷時檔案完整路徑（例如 C:/.../Northridge_NS.txt 或 .csv）：")
W_kip = float(input("請輸入結構重量 W (kips)："))
W_lbf = W_kip * 1000
m = W_lbf / 386.1  # slug

k_kip_in = float(input("請輸入結構勁度 k (kip/in)："))
k = k_kip_in * 1000  # lbf/in

zeta = float(input("請輸入阻尼比 ζ (0~1)："))
u0 = float(input("請輸入初始位移 u0 (in)："))
v0 = float(input("請輸入初始速度 v0 (in/s)："))
dt = float(input("請輸入時間間隔 Δt (s)："))
gamma = float(input("請輸入 Newmark 參數 δ (例如 0.5)："))  # δ → gamma
beta = float(input("請輸入 Newmark 參數 α (例如 0.25)："))  # α → beta

# === 結構參數 ===
wn = np.sqrt(k / m)
c = 2 * zeta * m * wn

# === 讀取地震歷時 ===
if file_path.endswith('.txt'):
    data = np.loadtxt(file_path)
elif file_path.endswith('.csv'):
    data = pd.read_csv(file_path).values
else:
    raise ValueError("不支援的檔案格式，請提供 .txt 或 .csv 檔案")

time = data[:, 0]
ag = data[:, 1] * 386.1  # g → in/s²
npts = len(ag)

# === 初始化 ===
u = np.zeros(npts)
v = np.zeros(npts)
a_resp = np.zeros(npts)

u[0] = u0
v[0] = v0
p = -m * ag  # 慣性力
a_resp[0] = (p[0] - c * v[0] - k * u[0]) / m

# === 預計算常數 ===
k_eff = k + gamma / (beta * dt) * c + m / (beta * dt ** 2)

# === 時間歷程計算 ===
for i in range(1, npts):
    # 有效力項
    rhs = p[i] + \
          m * ( (1 / (beta * dt ** 2)) * u[i - 1] + (1 / (beta * dt)) * v[i - 1] + (1 / (2 * beta) - 1) * a_resp[i - 1] ) + \
          c * ( (gamma / (beta * dt)) * u[i - 1] + ( (gamma / beta) - 1) * v[i - 1] + dt * ( (gamma / (2 * beta)) - 1) * a_resp[i - 1] )

    # 更新位移
    u[i] = rhs / k_eff

    # 更新加速度
    a_resp[i] = (1 / (beta * dt ** 2)) * (u[i] - u[i - 1] - dt * v[i - 1]) - (1 - 2 * beta) / (2 * beta) * a_resp[i - 1]

    # 更新速度
    v[i] = v[i - 1] + dt * ( (1 - gamma) * a_resp[i - 1] + gamma * a_resp[i] )

# === 計算最大值 ===
max_u = np.max(np.abs(u))
max_v = np.max(np.abs(v))
max_a = np.max(np.abs(a_resp))

# === 輸出到 CSV ===
save_path = input("請輸入要儲存結果的資料夾路徑：").strip()
filename = input("請輸入檔案名稱（不含副檔名）：").strip()
full_path = os.path.join(save_path, f"{filename}.csv")

df = pd.DataFrame({
    'Time (s)': time,
    'Displacement (in)': u,
    'Velocity (in/s)': v,
    'Acceleration (in/s^2)': a_resp
})

# 在最後一行加入最大值標註
max_row = pd.DataFrame({
    'Time (s)': ['MAX'],
    'Displacement (in)': [max_u],
    'Velocity (in/s)': [max_v],
    'Acceleration (in/s²)': [max_a]
})

df.to_csv(full_path, index=False)
print(f"結果已儲存至：{full_path}")
print(f'計算完成，結果已輸出至 {full_path}')
print(f'最大位移: {max_u:.4f} in')
print(f'最大速度: {max_v:.4f} in/s')
print(f'最大加速度: {max_a:.4f} in/s²')

disp_unit = "Displacement(inch)"
vel_unit = "Velocity(inch/s)"
acc_unit = "Acceleration(inch/s^2)"

# === 繪圖 ===
plt.figure(figsize=(10, 6))
plt.plot(time, u, color='black', linestyle='.', label='Displacement')
plt.title("Displacement vs Time")
plt.xlabel("Time (s)")
plt.ylabel(disp_unit)
plt.grid(True)
plt.savefig("displacement.png")
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(time, v, color='red', linestyle='.', label='Velocity')
plt.title("Velocity vs Time")
plt.xlabel("Time (s)")
plt.ylabel(vel_unit)
plt.grid(True)
plt.savefig("velocity.png")
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(time, a_resp, color='blue', linestyle='.', label='Acceleration')
plt.title("Acceleration vs Time")
plt.xlabel("Time (s)")
plt.ylabel(acc_unit)
plt.grid(True)
plt.savefig("acceleration.png")
plt.show()

