# from ML_analyze import *
from ML_analyze_adjust import *
import time
import numpy as np
import sys
import os

# Add parent directory to path so we can import from there
# (假設 american_put_pricer.py 在 ../pricing/ 路徑下)
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "pricing"))
from american_put_pricer import read_vol_data


def main():
    folder = "../data_process/data_pack"
    label = "post_vol_"
    
    # 1. 讀取訓練資料
    train_vol_data_path = f"../data_process/data_pack/{label}grid_train.npz"
    data, quote_dates, vol_surfaces, K_grid, T_grid = read_vol_data(train_vol_data_path, label)
    train_vol_surfaces = vol_surfaces

    # 2. 讀取測試資料
    test_vol_data_path = f"../data_process/data_pack/{label}grid_test.npz"
    # *** 修正 BUG 1: 這裡應該使用 test_vol_data_path ***
    test_data, test_quote_dates, test_vol_surfaces, _, _ = read_vol_data(test_vol_data_path, label)

    # 獲取 k_grid (從任一個 data 物件)
    k_grid = data["k_grid"]
    
    print(f"train_vol_surfaces shape: {train_vol_surfaces.shape}")
    print(f"test_vol_surfaces shape: {test_vol_surfaces.shape}")

    # 3. 合併
    all_vol_surfaces = np.concatenate([train_vol_surfaces, test_vol_surfaces], axis=0)
    print("all_vol_surfaces.shape", all_vol_surfaces.shape)
    
    # 4. 執行 SVD 分析
    svd_analysis(folder, all_vol_surfaces, k_grid, T_grid)


if __name__ == "__main__":
    # Time the execution of the main function
    start_time = time.time()

    # Run the main function from ML_analyze.py
    main()

    # Calculate and print execution time
    execution_time = time.time() - start_time
    print(f"Execution time: {execution_time:.2f} seconds")