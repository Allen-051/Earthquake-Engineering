import numpy as np
import matplotlib.pyplot as plt
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
        
        # 繪製圖表
        plt.figure(figsize=(10, 6))
        plt.plot(time, acc, color='blue', linestyle='-', label='Acceleration')
        plt.xlabel('time(s)')
        plt.ylabel('acceleration(g)')
        plt.title(f"{earthquake_name} Acceleration vs Time")
        plt.legend()
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

# 呼叫函數
plot_acc()