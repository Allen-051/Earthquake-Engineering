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

def matrix_1(m, u0, v0, k, zeta, dt, F):
    """計算參數"""
    # average acceleration method
    delta = 0.5
    alpha = 0.25
    c1 = 2 * zeta * math.sqrt(m * k)  # 阻尼係數

    beta_1 = m + c1 * delta * dt + k * alpha * dt**2

    # 起始加速度
    a0_1 = -c1/m * v0 - k /m * u0 + F / m
    
    # 定義A_1矩陣
    a11 = 1
    a12 = dt - k * alpha * dt**3/beta_1
    a13 = 0.5 * dt**2 - c1 * alpha *dt**3 / beta_1 - 0.5 * k * alpha * dt**4 / beta_1
    a21 = 0
    a22 = 1 - k * alpha * dt**2 / beta_1
    a23 = dt - c1 * alpha * dt**2 / beta_1 - k * alpha * dt**3 / (2 * beta_1)
    a31 = 0
    a32 = -k * dt / beta_1
    a33 = 1 - c1 * dt / beta_1 - k * dt**2 / (2 * beta_1)

    A_1 = np.array([[a11, a12, a13 ], [a21, a22, a23] ,[a31, a32, a33]])

    # 定義B_1向量
    b11 = alpha * dt**2 / beta_1
    b21 = delta * dt / beta_1
    b31 = 1 / beta_1

    B_1 = np.array([[b11], [b21], [b31]])

    #print("A_1矩陣")
    #print(pd.DataFrame(A_1).to_csv(index=False, header=False))
    #print("B_1向量：")
    #print(pd.DataFrame(B_1).to_csv(index=False, header=False))

    return A_1, B_1, a0_1

def displacement_loop(k1, k2, A_1, B_1, dt, u0, v0, a0_1, F, time_offset):
    """第一階段，找到u=1時的數據，並截斷後續資料"""
    steps = len(F) - 1
    x_hist = []
    x_j = np.array([[u0], [v0], [a0_1]])
    x_hist.append(x_j.copy())

    u_a = v_a = a_a = None
    found_a = False
    cut_idx = None

    for j in range(steps):
        delta_F = F[j+1] - F[j]
        x_j1 = np.matmul(A_1, x_j) + B_1 * delta_F
        x_hist.append(x_j1.copy())

        if not found_a and x_j1[0, 0] >= 1:
            u_a = x_j1[0, 0]
            v_a = x_j1[1, 0]
            a_a = x_j1[2, 0]
            found_a = True
            cut_idx = j + 1
            print(f"***a點***位移為1時：u={u_a:.4f}, v={v_a:.4f}, a={a_a:.4f}（將作為第二階段初始值）")
            break  # 只保留到u=1為止

        x_j = x_j1

    # 截斷資料
    if cut_idx is None:
        cut_idx = len(x_hist) - 1
    x_hist = x_hist[:cut_idx+1]
    x_hist = np.hstack(x_hist)
    # 修正：在這裡再計算 time
    time = np.arange(x_hist.shape[1]) * dt + time_offset
    x_time = np.vstack([time, x_hist])
    df = pd.DataFrame(x_time.T, columns=['time', 'u', 'v', 'a'])
    print("x_hist (第一階段):")
    print(df.to_string(index=False))

    # 計算此時結構恢復力Fs
    Fs = k1 * 1 + k2 * (u_a - 1)
    print(f"結構物恢復力 Fs: {Fs:.4f} (kip)\n") 

    return u_a, v_a, a_a, df, Fs

def matrix_2(m, u_a, v_a, k, zeta, dt, F, Fs):
    """計算第二階段參數與初始加速度"""
    delta = 0.5
    alpha = 0.25
    c2 = 2 * zeta * math.sqrt(abs(m * k))  # 阻尼係數
    beta_2 = m + c2 * delta * dt + k * alpha * dt**2

    # 修正第二階段初始加速度  
    a0_2 = - c2 / m * v_a - Fs / m  + F / m
    # 定義A_2矩陣
    a11 = 1
    a12 = dt - k * alpha * dt**3/beta_2
    a13 = 0.5 * dt**2 - c2 * alpha *dt**3 / beta_2 - 0.5 * k * alpha * dt**4 / beta_2
    a21 = 0
    a22 = 1 - k * alpha * dt**2 / beta_2
    a23 = dt - c2 * alpha * dt**2 / beta_2 - k * alpha * dt**3 / (2 * beta_2)
    a31 = 0
    a32 = -k * dt / beta_2
    a33 = 1 - c2 * dt / beta_2 - k * dt**2 / (2 * beta_2)

    A_2 = np.array([[a11, a12, a13 ], [a21, a22, a23] ,[a31, a32, a33]])

    # 定義B_3向量
    b11 = alpha * dt**2 / beta_2
    b21 = delta * dt / beta_2
    b31 = 1 / beta_2
    B_2 = np.array([[b11], [b21], [b31]])

    #print("A_2矩陣")
    #print(pd.DataFrame(A_2).to_csv(index=False, header=False))
    #print("B_2向量：")
    #print(pd.DataFrame(B_2).to_csv(index=False, header=False))
    return A_2, B_2, a0_2

def displacement_loop_2(k1, k2, A_2, B_2, dt, u_a, v_a, a_a, F, Fs ,time_offset):
    """第二階段，找到v由正變負時的數據，並截斷後續資料"""
    steps = len(F) - 1
    x_hist = []
    x_j = np.array([[u_a], [v_a], [a_a]])
    x_hist.append(x_j.copy())
    
    u_b = v_b = a_b = None
    found_b = False
    cut_idx = None

    for j in range(steps):
        delta_F = F[j+1] - F[j]
        x_j1 = np.matmul(A_2, x_j) + B_2 * delta_F
        x_hist.append(x_j1.copy())

        if not found_b and x_j[1, 0] > 0 and x_j1[1, 0] <= 0:
            # 取v剛變號的那個點（x_j1），也可插值
            u_b = x_j1[0, 0]
            v_b = x_j1[1, 0]
            a_b = x_j1[2, 0]
            found_b = True
            cut_idx = j + 1
            print(f"\n***b點***速度為0時，u={u_b:.4f}, v={v_b:.4f}, a={a_b:.4f}(將做為第三階段初始值)")
            break  # 只保留到v變號為止

        x_j = x_j1
    if cut_idx is None:
        cut_idx = len(x_hist) - 1
    x_hist = x_hist[:cut_idx+1]
    x_hist = np.hstack(x_hist)
    time = np.arange(x_hist.shape[1]) * dt + time_offset
    x_time = np.vstack([time, x_hist])
    df = pd.DataFrame(x_time.T, columns=['time', 'u', 'v', 'a'])
    print("x_hist (第二階段):")
    print(df.to_string(index=False))
    # 計算此時結構恢復力Fs
    Fs = Fs + k2 * (u_b - u_a)
    print(f"結構物恢復力 Fs: {Fs:.4f} (kip)\n") 
    return u_b, v_b, a_b, df, Fs

def matrix_3(m, u_b, v_b, k, zeta, dt, F, Fs):
    """計算第三階段參數與初始加速度"""
    # k = -k
    delta = 0.5
    alpha = 0.25
    c3 = 2 * zeta * math.sqrt(abs(m * k))  # 阻尼係數
    beta_3 = m + c3 * delta * dt + k * alpha * dt**2
    
    # 修正第三階段初始加速度
    a0_3 = -c3/m * v_b - Fs / m + F / m
    # 定義A_3矩陣
    a11 = 1
    a12 = dt - k * alpha * dt**3/beta_3
    a13 = 0.5 * dt**2 - c3 * alpha *dt**3 / beta_3 - 0.5 * k * alpha * dt**4 / beta_3
    a21 = 0
    a22 = 1 - k * alpha * dt**2 / beta_3
    a23 = dt - c3 * alpha * dt**2 / beta_3 - k * alpha * dt**3 / (2 * beta_3)
    a31 = 0
    a32 = -k * dt / beta_3
    a33 = 1 - c3 * dt / beta_3 - k * dt**2 / (2 * beta_3)

    A_3 = np.array([[a11, a12, a13 ], [a21, a22, a23] ,[a31, a32, a33]])

    # 定義B_3向量
    b11 = alpha * dt**2 / beta_3
    b21 = delta * dt / beta_3
    b31 = 1 / beta_3

    B_3 = np.array([[b11], [b21], [b31]])

    #print(pd.DataFrame(A_3).to_csv(index=False, header=False))
    #print("B_3向量：")
    #print(pd.DataFrame(B_3).to_csv(index=False, header=False))
    #print(f"第三階段初始加速度 a0_3: {a0_3:.4f}")
    return A_3, B_3, a0_3

def displacement_loop_3(k1, k2, A_3, B_3, dt, u_b, v_b, a_b, F, Fs, time_offset):
    """第三階段，從b點開始完整計算"""
    steps = len(F) - 1
    x_hist = []
    x_j = np.array([[u_b], [v_b], [a_b]])
    x_hist.append(x_j.copy())
    
    u_c = v_c = a_c = None
    found_c = False
    cut_idx = None
    u_c_control = u_b -2

    for j in range(steps):
        delta_F = F[j+1] - F[j]
        x_j1 = np.matmul(A_3, x_j) + B_3 * delta_F
        x_hist.append(x_j1.copy())

        if not found_c and (x_j[0, 0] > - u_c_control) >= 0 and (x_j1[0, 0]-u_c_control) <= 0:
            # 取v剛變號的那個點（x_j1），也可插值
            u_c = x_j1[0, 0]
            v_c = x_j1[1, 0]
            a_c = x_j1[2, 0]
            found_c = True
            cut_idx = j + 1
            print(f"\n***c點***位移U_max-2Xy時，u={u_c:.4f}, v={v_c:.4f}, a={a_c:.4f}(將做第四階段初始值)")
            break 

        x_j = x_j1

    if cut_idx is None:
        cut_idx = len(x_hist) - 1
    x_hist = x_hist[:cut_idx+1]
    x_hist = np.hstack(x_hist)
    time = np.arange(x_hist.shape[1]) * dt + time_offset
    x_time = np.vstack([time, x_hist])
    df = pd.DataFrame(x_time.T, columns=['time', 'u', 'v', 'a'])
    print("x_hist (第三階段):")
    print(df.to_string(index=False))
    # 計算結構物恢復力Fs
    Fs = k1 * 1 + k2 * (u_b - 1) + k1 * (u_c - u_b)
    print(f"結構物恢復力 Fs: {Fs:.4f} (kip)\n") 
    return u_c, v_c, a_c, df, Fs

def matrix_4(m, u_c, v_c, k, zeta, dt, F, Fs):
    """計算第四階段參數與初始加速度"""
    # k = -k
    delta = 0.5
    alpha = 0.25
    c4 = 2 * zeta * math.sqrt(abs(m * k))  # 阻尼係數
    beta_4 = m + c4 * delta * dt + k * alpha * dt**2

    a0_4 = -c4/m * v_c - Fs / m + F / m
    # 定義A_4矩陣
    a11 = 1
    a12 = dt - k * alpha * dt**3/beta_4
    a13 = 0.5 * dt**2 - c4 * alpha *dt**3 / beta_4 - 0.5 * k * alpha * dt**4 / beta_4
    a21 = 0
    a22 = 1 - k * alpha * dt**2 / beta_4
    a23 = dt - c4 * alpha * dt**2 / beta_4 - k * alpha * dt**3 / (2 * beta_4)
    a31 = 0
    a32 = -k * dt / beta_4
    a33 = 1 - c4 * dt / beta_4 - k * dt**2 / (2 * beta_4)

    A_4 = np.array([[a11, a12, -a13 ], [a21, a22, -a23] ,[a31, a32, a33]])

    # 定義B_4向量
    b11 = alpha * dt**2 / beta_4
    b21 = delta * dt / beta_4
    b31 = 1 / beta_4

    B_4 = np.array([[b11], [b21], [b31]])

    #print("A_4矩陣")
    #print(pd.DataFrame(A_4).to_csv(index=False, header=False))
    #print("B_4向量：")
    #print(pd.DataFrame(B_4).to_csv(index=False, header=False))
    #print(f"第四階段初始加速度 a0_4: {a0_4:.4f}")
    return A_4, B_4, a0_4

def displacement_loop_4(k1, k2, A_4, B_4, dt, u_c, v_c, a_c, F, Fs, time_offset):
    """第四階段，從c點開始完整計算或根據控制條件截斷"""
    steps = len(F) - 1
    x_hist = []
    x_j = np.array([[u_c], [v_c], [a_c]])
    x_hist.append(x_j.copy())

    u_d = v_d = a_d = None
    found_d = False
    cut_idx = None

    # 這裡可根據需求加入新的控制條件
    for j in range(steps):
        delta_F = F[j+1] - F[j]
        x_j1 = np.matmul(A_4, x_j) + B_4 * delta_F
        x_hist.append(x_j1.copy())

        if not found_d and x_j[1, 0] <= 0 and x_j1[1, 0] >= 0:
            # 取v剛變號的那個點（x_j1），也可插值
            u_d = x_j1[0, 0]
            v_d = x_j1[1, 0]
            a_d = x_j1[2, 0]
            found_d = True
            cut_idx = j + 1
            print(f"\n***d點***速度為0時，u={u_d:.4f}, v={v_d:.4f}, a={a_d:.4f}(將做為第五階段初始值)")
            
            break
        x_j = x_j1
    if cut_idx is None:
        cut_idx = len(x_hist) - 1

    x_hist = x_hist[:cut_idx+1]
    x_hist = np.hstack(x_hist)
    time = np.arange(x_hist.shape[1]) * dt + time_offset
    x_time = np.vstack([time, x_hist])
    df = pd.DataFrame(x_time.T, columns=['time', 'u', 'v', 'a'])
    print("x_hist (第四階段):")
    print(df.to_string(index=False))
    # 計算結構物恢復力Fs
    Fs =  Fs + k2 * (u_d - u_c)
    print(f"結構物恢復力 Fs: {Fs:.4f} (kip)\n") 
    return u_d, v_d, a_d, df, Fs

def matrix_5(m, u_d, v_d, k, zeta, dt, F, Fs):
    """計算第五階段參數與初始加速度""" 
    delta = 0.5
    alpha = 0.25
    c5 = 2 * zeta * math.sqrt(abs(m * k))  # 阻尼係數
    beta_5 = m + c5 * delta * dt + k * alpha * dt**2

    a0_5 = - c5/m * v_d - Fs / m + F / m
    # 定義A_5矩陣
    a11 = 1
    a12 = dt - k * alpha * dt**3/beta_5
    a13 = 0.5 * dt**2 - c5 * alpha *dt**3 / beta_5 - 0.5 * k * alpha * dt**4 / beta_5
    a21 = 0
    a22 = 1 - k * alpha * dt**2 / beta_5
    a23 = dt - c5 * alpha * dt**2 / beta_5 - k * alpha * dt**3 / (2 * beta_5)
    a31 = 0
    a32 = -k * dt / beta_5
    a33 = 1 - c5 * dt / beta_5 - k * dt**2 / (2 * beta_5)

    A_5 = np.array([[a11, a12, a13 ], [a21, a22, a23] ,[a31, a32, a33]])

    # 定義B_4向量
    b11 = alpha * dt**2 / beta_5
    b21 = delta * dt / beta_5
    b31 = 1 / beta_5

    B_5 = np.array([[b11], [b21], [b31]])

    #rint("A_5矩陣")
    #print(pd.DataFrame(A_5).to_csv(index=False, header=False))
    #print("B_5向量：")
    #print(pd.DataFrame(B_5).to_csv(index=False, header=False))
    #print(f"第五階段初始加速度 a0_5: {a0_5:.4f}")
    return A_5, B_5, a0_5

def displacement_loop_5(k1, k2, A_5, B_5, dt, u_d, v_d, a_d, F, Fs, time_offset):
    """第五階段，從d點開始完整計算或根據控制條件截斷"""
    steps = len(F) - 1
    x_hist = []
    x_j = np.array([[u_d], [v_d], [a_d]])
    x_hist.append(x_j.copy())
    
    u_e = v_e = a_e = None
    found_e = False
    cut_idx = None
    u_e_control = u_d + 1.6

    for j in range(steps):
        delta_F = F[j+1] - F[j]
        x_j1 = np.matmul(A_5, x_j) + B_5 * delta_F
        x_hist.append(x_j1.copy())

        if not found_e and (x_j[0, 0] - u_e_control) <= 0 and (x_j1[0, 0] - u_e_control) >= 0:
            u_e = x_j1[0, 0]
            v_e = x_j1[1, 0]
            a_e = x_j1[2, 0]
            found_e = True
            cut_idx = j + 1
            print(f"\n***e點***位移為-U_max+2Xy時，u={u_e:.4f}, v={v_e:.4f}, a={a_e:.4f}(將做為第六階段初始值)")
            
            break
        x_j = x_j1

    if cut_idx is None:
        cut_idx = len(x_hist) - 1

    x_hist = x_hist[:cut_idx+1]
    x_hist = np.hstack(x_hist)
    time = np.arange(x_hist.shape[1]) * dt + time_offset
    x_time = np.vstack([time, x_hist])
    df = pd.DataFrame(x_time.T, columns=['time', 'u', 'v', 'a'])
    print("x_hist (第五階段):")
    print(df.to_string(index=False))
    # 計算結構物恢復力Fs
    Fs =  Fs + k1 * (u_e - u_d)
    print(f"結構物恢復力 Fs: {Fs:.4f} (kip)\n") 
    return u_e, v_e, a_e, df, Fs

def matrix_6(m, u_e, v_e, k, zeta, dt, F0, Fs):
    """計算第六階段參數與初始加速度"""
    delta = 0.5
    alpha = 0.25
    c6 = 2 * zeta * math.sqrt(abs(m * k))  # 阻尼係數
    beta_6 = m + c6 * delta * dt + k * alpha * dt**2

    a0_6 = -c6/m * v_e - Fs / m + F0 / m
    # 定義A_6矩陣
    a11 = 1
    a12 = dt - k * alpha * dt**3 / beta_6
    a13 = 0.5 * dt**2 - c6 * alpha * dt**3 / beta_6 - 0.5 * k * alpha * dt**4 / beta_6
    a21 = 0
    a22 = 1 - k * alpha * dt**2 / beta_6
    a23 = dt - c6 * alpha * dt**2 / beta_6 - k * alpha * dt**3 / (2 * beta_6)
    a31 = 0
    a32 = -k * dt / beta_6
    a33 = 1 - c6 * dt / beta_6 - k * dt**2 / (2 * beta_6)

    A_6 = np.array([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]])

    # 定義B_6向量
    b11 = alpha * dt**2 / beta_6
    b21 = delta * dt / beta_6
    b31 = 1 / beta_6

    B_6 = np.array([[b11], [b21], [b31]])

    #print("A_6矩陣")
    #print(pd.DataFrame(A_6).to_csv(index=False, header=False))
    #print("B_6向量：")
    #print(pd.DataFrame(B_6).to_csv(index=False, header=False))
    #print(f"第六階段初始加速度 a0_6: {a0_6:.4f}")
    return A_6, B_6, a0_6

def displacement_loop_6(k1, k2, A_6, B_6, dt, u_e, v_e, a_e, F, Fs, time_offset):
    """第六階段，從e點開始根據控制條件截斷"""
    steps = len(F) - 1
    x_hist = []
    x_j = np.array([[u_e], [v_e], [a_e]])
    x_hist.append(x_j.copy())

    u_f = v_f = a_f = None
    found_f = False
    cut_idx = None

    for j in range(steps):
        delta_F = F[j+1] - F[j]
        x_j1 = np.matmul(A_6, x_j) + B_6 * delta_F
        x_hist.append(x_j1.copy())

        if not found_f and x_j[0, 0] <= 2 and x_j1[0, 0] >= 2:
            u_f = x_j1[0, 0]
            v_f = x_j1[1, 0]
            a_f = x_j1[2, 0]
            found_f = True
            cut_idx = j + 1
            print(f"\n***f點***位移回到Xy時，u={u_f:.4f}, v={v_f:.4f}, a={a_f:.4f}(分析中止)")
            
            break
        x_j = x_j1
    if cut_idx is None:
        cut_idx = len(x_hist) - 1

    x_hist = np.hstack(x_hist)
    time = np.arange(x_hist.shape[1]) * dt + time_offset
    x_time = np.vstack([time, x_hist])
    df = pd.DataFrame(x_time.T, columns=['time', 'u', 'v', 'a'])
    print("x_hist (第六階段):")
    print(df.to_string(index=False))
    # 計算結構物恢復力Fs
    Fs = Fs + k2 * (u_f - u_e)
    print(f"結構物恢復力 Fs: {Fs:.4f} (kip)\n") 
    return df, Fs

def calc_Fs_by_stage(df_all, df1, df2, df3, df4, df5, df6, k1, k2):
    # 計算每個階段的長度
    n1 = len(df1)
    n2 = len(df2)
    n3 = len(df3)
    n4 = len(df4)
    n5 = len(df5)
    n6 = len(df6)
    # 建立一個與 df_all 長度相同的 k_array
    k_array = np.zeros(len(df_all))
    # 第一階段用 k1
    k_array[:n1] = k1
    # 第二階段用 k2
    k_array[n1:n1+n2] = k2
    # 第三階段用 k1
    k_array[n1+n2:n1+n2+n3] = -k1
    # 第四階段用 k2
    k_array[n1+n2+n3:n1+n2+n3+n4] = -k2
    # 第五階段用 k1
    k_array[n1+n2+n3+n4:n1+n2+n3+n4+n5] = k1
    # 第六階段用 k2
    k_array[n1+n2+n3+n4+n5:] = k2

    # 計算 Fs
    df_all['Fs'] = k_array * df_all['u']
    return df_all

def plot_figures(df_all, k, save_path, filename):
    # 計算 Fs
    df_all['Fs'] = k * df_all['u']

    # 第一張：time-u
    plt.figure()
    plt.plot(df_all['time'], df_all['u'])
    plt.xlabel('Time (s)')
    plt.ylabel('Displacement u (inch)')
    plt.title('Time vs Displacement')
    plt.grid(False)
    plt.savefig(os.path.join(save_path, f"{filename}_fig1_u_t.png"))
    plt.close()
    plt.show()

    # 第二張：time-v
    plt.figure()
    plt.plot(df_all['time'], df_all['v'])
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity v (inch/s)')
    plt.title('Time vs Velocity')
    plt.grid(False)
    plt.savefig(os.path.join(save_path, f"{filename}_fig2_v_t.png"))
    plt.close()
    plt.show()

    # 第三張：time-a
    plt.figure()
    plt.plot(df_all['time'], df_all['a'])
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration a (inch/s²)')
    plt.title('Time vs Acceleration')
    plt.grid(False)
    plt.savefig(os.path.join(save_path, f"{filename}_fig3_a_t.png"))
    plt.close()
    plt.show()

    # 第四張：time-Fs
    plt.figure()
    plt.plot(df_all['time'], df_all['Fs'])
    plt.xlabel('Time (s)')
    plt.ylabel('Fs (kip)')
    plt.title('Time vs Fs')
    plt.grid(False)
    plt.savefig(os.path.join(save_path, f"{filename}_fig4_Fs_t.png"))
    plt.close()
    plt.show()

    # 第五張：u-Fs
    plt.figure()
    plt.plot(df_all['u'], df_all['Fs'])
    plt.xlabel('Displacement u (in)')
    plt.ylabel('Fs (kip)')
    plt.title('Displacement vs Fs')
    plt.grid(False)
    plt.savefig(os.path.join(save_path, f"{filename}_fig5_Fs_u.png"))
    plt.close()
    plt.show()

def main():
    save_path, filename, m, k1, k2, zeta, u0, v0, dt, F0 = get_user_input()
    A_1, B_1, a0_1 = matrix_1(m, u0, v0, k1, zeta, dt, F0)
    F = np.zeros(1001)
    F[0] = F0
    for i in range(1, 1001):
        F[i] = F[i-1]

    # 第一階段
    u_a, v_a, a_a, df1, Fs1 = displacement_loop(k1, k2, A_1, B_1, dt, u0, v0, a0_1, F, time_offset=0.0)
    t1_end = df1['time'].iloc[-1]

    # 第二階段
    A_2, B_2, a0_2 = matrix_2(m, u_a, v_a, k2, zeta, dt, F0, Fs1)
    u_b, v_b, a_b, df2, Fs2 = displacement_loop_2(k1, k2, A_2, B_2, dt, u_a, v_a, a0_2, F, Fs1, time_offset=t1_end)
    t2_end = df2['time'].iloc[-1]

    # 第三階段
    A_3, B_3, a0_3 = matrix_3(m, u_b, v_b, k1, zeta, dt, F0, Fs2)
    u_c, v_c, a_c, df3, Fs3 = displacement_loop_3(k1, k2, A_3, B_3, dt, u_b, v_b, a0_3, F, Fs2, time_offset=t2_end)
    t3_end = df3['time'].iloc[-1]

    # 第四階段
    A_4, B_4, a0_4 = matrix_4(m, u_c, v_c, k2, zeta, dt, F0, Fs3)
    u_d, v_d, a_d, df4, Fs4 = displacement_loop_4(k1, k2, A_4, B_4, dt, u_c, v_c, a0_4, F, Fs3, time_offset=t3_end)
    t4_end = df4['time'].iloc[-1]

    # 第五階段
    A_5, B_5, a0_5 = matrix_5(m, u_d, v_d, k1, zeta, dt, F0, Fs4)
    u_e, v_e, a_e, df5, Fs5 = displacement_loop_5(k1, k2, A_5, B_5, dt, u_d, v_d, a0_5, F, Fs4, time_offset=t4_end)
    t5_end = df5['time'].iloc[-1]
    # 第六階段
    A_6, B_6, a0_6 = matrix_6(m, u_e, v_e, k2, zeta, dt, F0, Fs5)
    df6, Fs6 = displacement_loop_6(k1, k2, A_6, B_6, dt, u_e, v_e, a0_5, F, Fs5, time_offset=t5_end)


    print("\n=== 第一階段 DataFrame ===")
    print(df1.to_string(index=False))
    print("\n=== 第二階段 DataFrame ===")
    print(df2.to_string(index=False))
    print("\n=== 第三階段 DataFrame ===")
    print(df3.to_string(index=False))
    print("\n=== 第四階段 DataFrame ===")
    print(df4.to_string(index=False))
    print("\n=== 第五階段 DataFrame ===")
    print(df5.to_string(index=False))
    print("\n=== 第六階段 DataFrame ===")
    print(df6.to_string(index=False))

    df_all = pd.concat([df1, df2, df3, df4, df5, df6], ignore_index=True)
    df_all = calc_Fs_by_stage(df_all, df1, df2, df3, df4, df5, df6, k1, k2)
    # 繪圖
    plot_figures(df_all, k1, save_path, filename)

    # 輸出成一個 csv
    df_all.to_csv(os.path.join(save_path, f"{filename}_full_result.csv"), index=False)
    print(f"\n所有階段已整合儲存至：{os.path.join(save_path, f'{filename}_all.csv')}")


if __name__ == "__main__":
    main()