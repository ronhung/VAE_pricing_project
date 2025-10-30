import numpy as np

# 根據您的 main_VAE.py, folder 路徑如下
folder = "../data_process/data_pack"
vae_test_loss_path = f"{folder}/test_losses.npy"

# 加載 VAE 的測試損失歷史紀錄
try:
    vae_losses = np.load(vae_test_loss_path)
    
    # 獲取最後一個 epoch 的 test_loss
    final_vae_test_loss = vae_losses[-1]
    
    print(f"成功加載 VAE 損失檔案: {vae_test_loss_path}")
    print(f"VAE 總共訓練了 {len(vae_losses)} 個 epochs。")
    print(f"VAE 最終的 test_loss 是： {final_vae_test_loss}")

except FileNotFoundError:
    print(f"錯誤：找不到 VAE 損失檔案： {vae_test_loss_path}")
    print("您確定您已經執行過 main_VAE.py 中的 train_and_save_VAE_alone 嗎？")