import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import os
# 設定 matplotlib 字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

def read_earthquake_csv():
    """
    讀取地震加速度資料
    """
    # 詢問使用者輸入檔案路徑
    while True:
        file_path = input("請輸入地震加速度資料的 CSV 檔案路徑: ").strip()
        if os.path.isfile(file_path):
            break
        else:
            print("找不到檔案，請重新輸入正確的路徑！")

    data = pd.read_csv(file_path, header=None)
    time = data.iloc[:, 0].values
    ground_acceleration = data.iloc[:, 1].values

    # 詢問使用者儲存檔案路徑，若不存在則自動建立
    save_path = input("請輸入結果儲存路徑（直接按Enter則預設為桌面）: ").strip()
    if save_path == "":
        save_path = os.path.join(os.path.expanduser("~"), "Desktop")
        print(f"未輸入路徑，預設儲存於：{save_path}")
    if not os.path.exists(save_path):
        try:
            os.makedirs(save_path)
        except Exception as e:
            print(f"無法建立資料夾：{e}，請重新輸入。")
            return read_earthquake_csv()  # 重新詢問

    return time, ground_acceleration, save_path


def compute_modal_shapes(M, K):
    eigvals, eigvecs = eigh(K, M)
    omega_n = np.sqrt(eigvals)
    normalized_modes = eigvecs / np.max(np.abs(eigvecs), axis=0)
    frequencies = omega_n / (2 * np.pi)
    return normalized_modes, frequencies, omega_n

def system_matrices_nodamp():
    """產生無TMD的3階主結構系統矩陣"""
    m1 = m2 = m3 = 8.46e7
    k1 = k2 = k3 = 71200000
    c1 = c2 = c3 = 1552226.752869

    M = np.diag([m1, m2, m3])
    K = np.array([
        [k1 + k2, -k2,     0],
        [-k2,     k2 + k3, -k3],
        [0,      -k3,      k3]
    ])
    C = np.array([
        [c1 + c2, -c2,     0],
        [-c2,     c2 + c3, -c3],
        [0,      -c3,      c3]
    ])
    return M, K, C

def system_matrices_single_tmd(tmd_idx, md_r_list, omega_r_list, zeta_r_list):
    
    m1 = m2 = m3 = 8.46e7
    k1 = k2 = k3 = 71200000
    c1 = c2 = c3 = 1552226.752869

    # TMD參數
    md_r = md_r_list[tmd_idx]
    omega_r = omega_r_list[tmd_idx]
    zeta_r = zeta_r_list[tmd_idx]
    m_main = [m1, m2, m3][tmd_idx]
    k_main = [k1, k2, k3][tmd_idx]

    md = md_r * m_main
    omegad = omega_r * np.sqrt(k_main / m_main)
    Cd = zeta_r * (2 * np.sqrt(k_main / m_main))
    kd = md * omegad ** 2
    cd = Cd * md

    # 4自由度系統
    M = np.diag([m1, m2, m3, md])
    K = np.zeros((4, 4))
    C = np.zeros((4, 4))
    # 主結構部分
    K[:3, :3] = np.array([
        [k1 + k2, -k2,     0],
        [-k2,     k2 + k3, -k3],
        [0,      -k3,      k3]
    ])
    C[:3, :3] = np.array([
        [c1 + c2, -c2,     0],
        [-c2,     c2 + c3, -c3],
        [0,      -c3,      c3]
    ])
    # TMD耦合
    K[tmd_idx, tmd_idx] += kd
    K[tmd_idx, 3] = -kd
    K[3, tmd_idx] = -kd
    K[3, 3] = kd

    C[tmd_idx, tmd_idx] += cd
    C[tmd_idx, 3] = -cd
    C[3, tmd_idx] = -cd
    C[3, 3] = cd

    return M, K, C

def newmark_linear_acceleration(M, K, C, ag, dt):
    dof = M.shape[0]
    N = len(ag)
    e = np.ones(dof)
    F = np.array([-M @ e * ag[i] for i in range(N)]).T

    U = np.zeros((dof, N))
    V = np.zeros((dof, N))
    A = np.zeros((dof, N))

    beta, gamma = 1 / 6, 1 / 2
    A[:, 0] = np.linalg.inv(M) @ (F[:, 0] - C @ V[:, 0] - K @ U[:, 0])
    K_eff = K + gamma / (beta * dt) * C + M / (beta * dt ** 2)
    K_eff_inv = np.linalg.inv(K_eff)

    for i in range(1, N):
        delta_F = (
            F[:, i]
            + M @ (1 / (beta * dt ** 2) * U[:, i - 1] + 1 / (beta * dt) * V[:, i - 1] + (1 / (2 * beta) - 1) * A[:, i - 1])
            + C @ (gamma / (beta * dt) * U[:, i - 1] + (gamma / beta - 1) * V[:, i - 1] + dt * (gamma / (2 * beta) - 1) * A[:, i - 1])
        )
        U[:, i] = K_eff_inv @ delta_F
        A[:, i] = 1 / (beta * dt ** 2) * (U[:, i] - U[:, i - 1]) - 1 / (beta * dt) * V[:, i - 1] - (1 / (2 * beta) - 1) * A[:, i - 1]
        V[:, i] = V[:, i - 1] + dt * ((1 - gamma) * A[:, i - 1] + gamma * A[:, i])

    return U

def peak_mean_RMS_block(u_nodamp, u_tmd, floor_label, file=None, u_tmd_disp=None, tmd_num=None):
    """格式化輸出單一樓層的原結構與加TMD結果及相對誤差，並可輸出TMD本體位移統計"""
    # 原結構
    peak_n = np.max(np.abs(u_nodamp))
    mean_n = np.mean(u_nodamp)
    rms_n = np.sqrt(np.mean(u_nodamp ** 2))
    # 加TMD
    peak_t = np.max(np.abs(u_tmd))
    mean_t = np.mean(u_tmd)
    rms_t = np.sqrt(np.mean(u_tmd ** 2))
    # 相對誤差
    err_peak = (peak_t - peak_n) / peak_n * 100 if peak_n != 0 else np.nan
    err_mean = (mean_t - mean_n) / mean_n * 100 if mean_n != 0 else np.nan
    err_rms = (rms_t - rms_n) / rms_n * 100 if rms_n != 0 else np.nan

    lines = []
    if tmd_num is not None and u_tmd_disp is not None:
        # TMD本體位移統計
        peak_tmd = np.max(np.abs(u_tmd_disp))
        mean_tmd = np.mean(u_tmd_disp)
        rms_tmd = np.sqrt(np.mean(u_tmd_disp ** 2))
        lines.append(f"TMD_{tmd_num}號 阻尼器位移反應：")
        lines.append(f"  PEAK = {peak_tmd:.6e} m")
        lines.append(f"  MEAN = {mean_tmd:.6e} m")
        lines.append(f"  RMS  = {rms_tmd:.6e} m\n")

    lines.append(f"***{floor_label}位移反應***")
    lines.append("原結構 結果:")
    lines.append(f"  PEAK = {peak_n:.6e} m")
    lines.append(f"  MEAN = {mean_n:.6e} m")
    lines.append(f"  RMS  = {rms_n:.6e} m")
    lines.append("加裝TMD 結果:")
    lines.append(f"  PEAK = {peak_t:.6e} m")
    lines.append(f"  MEAN = {mean_t:.6e} m")
    lines.append(f"  RMS  = {rms_t:.6e} m\n")
    lines.append(f" *PEAK 相對誤差 = {err_peak:.2f} %")
    lines.append(f" *MEAN 相對誤差 = {err_mean:.2f} %")
    lines.append(f" *RMS  相對誤差 = {err_rms:.2f} %\n")
    # 輸出
    for line in lines:
        print(line)
    if file is not None:
        for line in lines:
            file.write(line + "\n")

def plot_graphs_by_floor(save_path, time, U_nodamp, U_damp, tmd_idx):
    floor_names = ["1F", "2F", "3F"]
    colors = ["blue", "green", "red"]
    tmd_num = [7, 8, 9][tmd_idx]
    for i in range(3):
        plt.figure(figsize=(10, 5))
        plt.plot(time, U_nodamp[i], label=f"{floor_names[i]} 原結構位移", color=colors[i], linestyle='-', linewidth=0.5)
        plt.plot(time, U_damp[i], label=f"{floor_names[i]} 有TMD位移", color=colors[i], linestyle='--', linewidth=1)
        plt.axhline(0, color='black', linewidth=0.5)  # 在y=0畫黑色細線
        plt.xlabel("Time (s)")
        plt.ylabel("Displacement (m)")
        plt.title(f"{floor_names[i]} 位移比較（TMD{tmd_num}）")
        plt.legend()
        plt.grid(False)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"u_d{tmd_num}_{i+1}F.jpg"), dpi=300)
        plt.close()

def plot_structure_and_tmd_disp(save_path, time, U_tmd, tmd_num):
    """
    畫出結構物一樓、二樓、三樓及單一TMD本體的位移反應（四條線）
    """
    floor_names = ["1F", "2F", "3F", f"TMD{tmd_num}"]
    colors = ["blue", "green", "red", "purple"]
    plt.figure(figsize=(10, 5))
    for i in range(3):
        plt.plot(time, U_tmd[i], label=f"{floor_names[i]} 位移", color=colors[i], linewidth=1)
    plt.plot(time, U_tmd[3], label=f"{floor_names[3]} 位移", color=colors[3], linewidth=1.5, linestyle="--")
    plt.axhline(0, color='black', linewidth=0.5)
    plt.xlabel("Time (s)")
    plt.ylabel("Displacement (m)")
    plt.title(f"{floor_names[3]} 加裝於結構物時各樓層與TMD位移比較")
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"u_structure_TMD{tmd_num}.jpg"), dpi=300)
    plt.close()

def plot_all_tmd_comparison(save_path, time, U_nodamp, U_tmd_list):
    floor_names = ["1F", "2F", "3F"]
    tmd_labels = ["TMD7", "TMD8", "TMD9"]
    colors = ["black", "blue", "green", "red"]
    linestyles = ["-", "dotted", "dotted", "dotted"]

    for i in range(3):
        plt.figure(figsize=(10, 5))
        plt.plot(time, U_nodamp[i], label="原結構", color=colors[0], linestyle=linestyles[0], linewidth=0.5)
        for j, U_tmd_main in enumerate(U_tmd_list):
            plt.plot(time, U_tmd_main[i], label=f"{tmd_labels[j]}", color=colors[j+1], linestyle=linestyles[j+1], linewidth=1)
        plt.axhline(0, color='black', linewidth=0.5)  # 在y=0畫黑色細線
        plt.xlabel("Time (s)")
        plt.ylabel("Displacement (m)")
        plt.title(f"{floor_names[i]} 各TMD配置位移比較")
        plt.legend()
        plt.grid(False)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"u_compare_{i+1}F.jpg"), dpi=300)
        plt.close()

def plot_tmd_disp_comparison(save_path, time, tmd_disp_list):

    tmd_labels = ["TMD7", "TMD8", "TMD9"]
    colors = ["blue", "green", "red"]
    plt.figure(figsize=(10, 5))
    for i, disp in enumerate(tmd_disp_list):
        plt.plot(time, disp, label=tmd_labels[i], color=colors[i], linewidth=1)
    plt.axhline(0, color='black', linewidth=0.5)  # 在y=0畫黑色細線
    plt.xlabel("Time (s)")
    plt.ylabel("TMD Displacement (m)")
    plt.title("三個阻尼器本體位移反應比較")
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "TMD_disp_comparison.jpg"), dpi=300)
    plt.close()


# 主程式流程
def main():
    # 讀取地震加速度資料
    time, ag, save_path = read_earthquake_csv()
    dt = time[1] - time[0]

    # TMD參數list
    md_r_list = [
        float(input("請輸入加裝於1號TMD質量比（如0.03）: ")),
        float(input("請輸入加裝於2F的TMD質量比（如0.1）: ")),
        float(input("請輸入加裝於3F的TMD質量比（如0.2）: "))
    ]
    omega_r_list = [0.9592, 0.8789, 0.7815]
    zeta_r_list = [0.0857, 0.1527, 0.2098]

    # 無阻尼結構
    M_nodamp, K_nodamp, C_nodamp = system_matrices_nodamp()
    U_nodamp = newmark_linear_acceleration(M_nodamp, K_nodamp, C_nodamp, ag, dt)

    # 原始結構模態資訊
    modal_shapes, frequencies, omega_n = compute_modal_shapes(M_nodamp, K_nodamp)
    modal_data = pd.DataFrame({
        "Mode": [f"Mode {i+1}" for i in range(len(modal_shapes))],
        "Frequency (Hz)": frequencies,
        "Period (s)": 1 / frequencies,
        "Omega_n (rad/s)": omega_n,
        "Modal Shape": [modal_shapes[:, i] for i in range(modal_shapes.shape[1])]
    })
    print("原始結構模態與頻率：")
    print(modal_data)

    # 收集 modal_info
    modal_info = []
    for i in range(len(frequencies)):
        modal_info.append({
            "Case": "Original",
            "Mode": f"Mode {i+1}",
            "Frequency (Hz)": frequencies[i],
            "Period (s)": 1 / frequencies[i],
            "Omega_n (rad/s)": omega_n[i]
        })

    txt_path = os.path.join(save_path, "peak_mean_RMS_RE.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        # 依序分析三種TMD配置
        for tmd_idx, floor_name in enumerate(["1F", "2F", "3F"]):
            tmd_num = [7, 8, 9][tmd_idx]
            M_tmd, K_tmd, C_tmd = system_matrices_single_tmd(tmd_idx, md_r_list, omega_r_list, zeta_r_list)
            U_tmd = newmark_linear_acceleration(M_tmd, K_tmd, C_tmd, ag, dt)
            U_tmd_main = U_tmd[:3, :]  # 只取主結構三樓層

            header = f"==========TMD_{tmd_num}號=========="
            print(header)
            f.write(header + "\n")
            for i, floor_label in enumerate(["一樓", "二樓", "三樓"]):
                # 只在第一層時輸出TMD本體位移統計，避免重複
                if i == 0:
                    peak_mean_RMS_block(U_nodamp[i], U_tmd_main[i], floor_label, file=f, u_tmd_disp=U_tmd[3, :], tmd_num=tmd_num)
                else:
                    peak_mean_RMS_block(U_nodamp[i], U_tmd_main[i], floor_label, file=f)

            # 加裝TMD後的模態與頻率
            modal_shapes_tmd, frequencies_tmd, omega_n_tmd = compute_modal_shapes(M_tmd, K_tmd)
            modal_data_tmd = pd.DataFrame({
                "Mode": [f"Mode {i+1}" for i in range(len(modal_shapes_tmd))],
                "Frequency (Hz)": frequencies_tmd,
                "Period (s)": 1 / frequencies_tmd,
                "Omega_n (rad/s)": omega_n_tmd,
                "Modal Shape": [modal_shapes_tmd[:, i] for i in range(modal_shapes_tmd.shape[1])]
            })
            print(f"\nTMD_{tmd_num}號加裝後結構物模態與頻率：")
            print(modal_data_tmd)

            # 收集 modal_info
            for i in range(len(frequencies_tmd)):
                modal_info.append({
                    "Case": f"TMD_{tmd_num}",
                    "Mode": f"Mode {i+1}",
                    "Frequency (Hz)": frequencies_tmd[i],
                    "Period (s)": 1 / frequencies_tmd[i],
                    "Omega_n (rad/s)": omega_n_tmd[i]
                })

            # 畫圖（傳入 tmd_idx 以正確命名）
            plot_structure_and_tmd_disp(save_path, time, U_tmd, tmd_num)

            # 儲存
            result_file = os.path.join(save_path, f"u_comparison_{floor_name}_TMD.csv")
            pd.DataFrame({
                "Time": time,
                "U1_nodamp": U_nodamp[0], "U1_tmd": U_tmd_main[0],
                "U2_nodamp": U_nodamp[1], "U2_tmd": U_tmd_main[1],
                "U3_nodamp": U_nodamp[2], "U3_tmd": U_tmd_main[2],
            }).to_csv(result_file, index=False)

        # 計算加裝阻尼器之後各樓層的位移反應
        U_tmd_list = []
        for tmd_idx in range(3):
            M_tmd, K_tmd, C_tmd = system_matrices_single_tmd(tmd_idx, md_r_list, omega_r_list, zeta_r_list)
            U_tmd = newmark_linear_acceleration(M_tmd, K_tmd, C_tmd, ag, dt)
            U_tmd_main = U_tmd[:3, :]
            U_tmd_list.append(U_tmd_main)
        # 計算阻尼器本身的位移反應
        tmd_disp_list = []
        for tmd_idx in range(3):
            M_tmd, K_tmd, C_tmd = system_matrices_single_tmd(tmd_idx, md_r_list, omega_r_list, zeta_r_list)
            U_tmd = newmark_linear_acceleration(M_tmd, K_tmd, C_tmd, ag, dt)
            tmd_disp_list.append(U_tmd[3, :])

        plot_all_tmd_comparison(save_path, time, U_nodamp, U_tmd_list)
        plot_tmd_disp_comparison(save_path, time, tmd_disp_list)

    # 匯出 modal_info.csv
    modal_info_df = pd.DataFrame(modal_info)
    modal_info_path = os.path.join(save_path, "modal_info.csv")
    modal_info_df.to_csv(modal_info_path, index=False)
    print(f"\n模態資訊已儲存至：{modal_info_path}")

if __name__ == "__main__":
    main()
