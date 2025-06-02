# 1.詢問用者地震歷時(csv或txt)
# 2.讀取歷時資料，第一行為時間，第二行為加速度(g)
# 3.計算位移反應，以線性加速度法(Newmark-beta法)計算
# 4.輸出結果，包含位移、速度、加速度的時間歷程
# 5.將結果繪製成圖
# 5.將結果儲存為csv檔案
# input_path = input("請輸入地震歷時檔案路徑: ")
# save_path = input("請輸入結果儲存路徑: ").strip()

import numpy as np
import pandas as pd
from scipy.linalg import eigh
import matplotlib.pyplot as plt
# from scipy.integrate import cumtrapz
import os

def plot_acc():
    # 請使用者輸入地震資料檔案路徑
    file_path = input("請輸入地震資料的 txt 檔案路徑：")
    earthquake_name = input("請輸入地震名稱：")
    
    try:
        # 讀取檔案內容
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        # 偵測關鍵字並排除標題行
        time = []
        acc = []
        for line in lines:
            if "Time(s)" in line or "Acc(g)" in line:
                continue
            # 分割數據並存入陣列
            data = line.strip().split()
            if len(data) == 2:  # 確保有兩列數據
                time.append(float(data[0]))
                acc.append(float(data[1]))
        
        # 將數據轉換為 numpy 陣列
        time = np.array(time)
        acc = np.array(acc)
        # 計算平均值、均方根植、尖峰值
        mean_acc = np.mean(acc)
        rms_acc = np.sqrt(np.mean(acc**2))
        peak_acc = np.max(np.abs(acc))
        print(f"平均加速度: {mean_acc:.4f} g")
        print(f"均方根加速度: {rms_acc:.4f} g")
        print(f"尖峰加速度: {peak_acc:.4f} g")
        # 建立圖例文字
        legend_label = (
        f"Acceleration\n"
        f"Mean: {mean_acc:.4f} g\n"
        f"RMS: {rms_acc:.4f} g\n"
        f"Peak: {peak_acc:.4f} g"
        )
        
        # 繪製圖表
        plt.figure(figsize=(10, 6))
        plt.plot(time, acc, color='black', linestyle='-', label=legend_label, linewidth=1)
        plt.xlabel('time(s)')
        plt.ylabel('acceleration(g)')
        plt.title(f"{earthquake_name} Acceleration Time History Data")
        plt.legend(loc='upper right')
        plt.grid(False)  # 不顯示格線
        plt.tight_layout()
        
        # 請使用者輸入儲存資料夾
        save_dir = input("請輸入要儲存圖表的資料夾路徑：")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 儲存圖表
        save_path = os.path.join(save_dir, "acceleration_plot.png")
        plt.savefig(save_path)
        print(f"圖表已儲存至：{save_path}")
        
        # 顯示圖表
        plt.show()
    
    except Exception as e:
        print(f"發生錯誤：{e}")

plot_acc()