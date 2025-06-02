import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

G_TO_M_S2 = 9.80665  # g → m/s²

def read_earthquake(file_path):
    df = pd.read_csv(file_path, sep=r'\s+', header=None, names=["Time", "Ag_g"])
    df["Ag"] = df["Ag_g"] * G_TO_M_S2
    return df["Time"].to_numpy(), df["Ag"].to_numpy()

def get_input_parameters(use_tmd):
    dof = 4 if use_tmd else 3
    m = np.zeros(dof)
    k = np.zeros(dof)
    c = np.zeros(dof)

    for i in range(3):
        print(f"\n--- 請輸入結構第 {i+1} 層自由度參數 ---")
        m[i] = float(input(f"質量 m{i+1} (kg): "))
        omega = float(input(f"自然頻率 ω{i+1} (rad/s): "))
        zeta = float(input(f"阻尼比 ζ{i+1}: "))
        k[i] = float(input(f"勁度 k{i+1} (N/m): "))
        c[i] = 2 * m[i] * omega * zeta

    if use_tmd:
        print(f"\n--- 請輸入 TMD（第4自由度）參數 ---")
        m[3] = float(input("TMD 質量 m_d (kg): "))
        omega = float(input("TMD 自然頻率 ω_d (rad/s): "))
        zeta = float(input("TMD 阻尼比 ζ_d: "))
        k[3] = float(input("TMD 勁度 k_d (N/m): "))
        c[3] = 2 * m[3] * omega * zeta

    return m, k, c

def build_coupled_matrices(m, k, c, use_tmd):
    dof = len(m)
    M = np.diag(m)
    K = np.zeros((dof, dof))
    C = np.zeros((dof, dof))

    for i in range(min(3, dof)):
        K[i, i] = k[i]
        C[i, i] = c[i]

    if use_tmd:
        K[2, 2] += k[3]
        K[2, 3] = -k[3]
        K[3, 2] = -k[3]
        K[3, 3] = k[3]

        C[2, 2] += c[3]
        C[2, 3] = -c[3]
        C[3, 2] = -c[3]
        C[3, 3] = c[3]

    return M, C, K

def newmark_linear_acceleration(M, C, K, ag, dt):
    N = len(ag)
    dof = M.shape[0]

    u = np.zeros((N, dof))
    v = np.zeros((N, dof))
    a = np.zeros((N, dof))

    invM = np.linalg.inv(M)
    a[0] = -invM @ (C @ v[0] + K @ u[0] + M @ np.ones(dof) * ag[0])

    beta = 1/6
    gamma = 1/2
    a0 = 1 / (beta * dt**2)
    a1 = gamma / (beta * dt)
    a2 = 1 / (beta * dt)
    a3 = 1 / (2 * beta) - 1
    a4 = gamma / beta - 1
    a5 = dt / 2 * (gamma / beta - 2)

    K_eff = K + a0 * M + a1 * C
    K_eff_inv = np.linalg.inv(K_eff)

    for i in range(1, N):
        dp = -M @ np.ones(dof) * (ag[i] - ag[i-1])
        dp_eff = (
            dp + M @ (a0 * u[i-1] + a2 * v[i-1] + a3 * a[i-1])
               + C @ (a1 * u[i-1] + a4 * v[i-1] + a5 * a[i-1])
        )
        u[i] = K_eff_inv @ dp_eff
        v[i] = a1 * (u[i] - u[i-1]) - a4 * v[i-1] - a5 * a[i-1]
        a[i] = a0 * (u[i] - u[i-1]) - a2 * v[i-1] - a3 * a[i-1]

    return u

def plot_displacement(time, u):
    plt.figure(figsize=(10, 6))
    for i in range(u.shape[1]):
        plt.plot(time, u[:, i], label=f"DOF {i+1}")
    plt.xlabel("Time (s)")
    plt.ylabel("Displacement (m)")
    plt.title("Displacement duration history")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def save_to_csv(time, u, filename="dispalcement.csv"):
    columns = [f"DOF {i+1} Displacement (m)" for i in range(u.shape[1])]
    df = pd.DataFrame(u, columns=columns)
    df.insert(0, "Time (s)", time)
    df.to_csv(filename, index=False)
    print(f"✅ 位移歷時資料已儲存為 {filename}")

def main():
    use_tmd = input("是否加裝 TMD？(y/n): ").lower() == 'y'
    file_path = input("請輸入地震歷時檔案路徑 (.txt，兩欄格式)：")
    time, ag = read_earthquake(file_path)
    dt = time[1] - time[0]

    m, k, c = get_input_parameters(use_tmd)
    M, C, K = build_coupled_matrices(m, k, c, use_tmd)

    u = newmark_linear_acceleration(M, C, K, ag, dt)

    plot_displacement(time, u)
    save_to_csv(time, u)

if __name__ == "__main__":
    main()
