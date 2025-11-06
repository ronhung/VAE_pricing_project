# -------------------------------------------------------------------
# main_HPO.py
#
# 使用 Optuna (貝葉斯優化) 來自動搜尋 
# TransformerAutoencoder (TAE) 的最佳超參數。
#
# 依賴於: Transformer_model.py, optuna
# -------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import os
import optuna # 導入 Optuna
from optuna.exceptions import TrialPruned # 用於剪枝
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split # 導入分割工具
# -------------------------------------------------------------------
# 1. 從我們現有的 Transformer_model.py 中導入構建模塊
#    (請確保此腳本與 Transformer_model.py 在同一資料夾中)
# -------------------------------------------------------------------
try:
    from Transformer_model import (
        TransformerAutoencoder, 
        create_dataloader
    )
except ImportError:
    print("錯誤: 找不到 Transformer_model.py")
    print("請確保 main_HPO.py 與 Transformer_model.py 放在同一個資料夾中。")
    exit(1)

# -------------------------------------------------------------------
# 2. 全局變量 (用於 HPO)
# -------------------------------------------------------------------
FOLDER = "../data_process/data_pack"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64     # HPO 時可以使用稍大的 Batch Size
N_EPOCHS_PER_TRIAL = 50 # 為了快速迭代，我們先用 50 epochs (之後可改回 100)
N_TRIALS = 500       # 總共要嘗試 50 種不同的超參數組合


# -------------------------------------------------------------------
# 3. Objective (目標函數)
#    Optuna 會不斷調用此函數，並試圖最小化它的回傳值。
# -------------------------------------------------------------------

def objective(trial: optuna.trial.Trial) -> float:
    """
    Optuna 的目標函數。
    1. 建議一組超參數。
    2. 建立並訓練 TAE 模型。
    3. 回傳最佳的 *validation_loss*。 (已修正 Data Snooping)
    """
    
    # --- A. 定義超參數的「搜尋空間」 ---
    # (這部分保持不變)
    
    # 1. 學習率 (log 尺度)
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    
    # 2. 架構參數
    d_model = trial.suggest_categorical("d_model", [32, 64, 128, 256])
    nhead = trial.suggest_categorical("nhead", [2, 4, 8])
    num_encoder_layers = trial.suggest_int("num_encoder_layers", 2, 6)
    num_decoder_layers = num_encoder_layers # 保持對稱
    dim_feedforward = trial.suggest_categorical("dim_feedforward", [64, 128, 256, 512])

    # 3. 檢查約束：nhead 必須能整除 d_model
    if d_model % nhead != 0:
        print(f"Pruning trial: d_model={d_model} % nhead={nhead} != 0")
        raise TrialPruned()

    print(f"\n--- [Trial {trial.number}] ---")
    print(f"Params: lr={lr:.2e}, d_model={d_model}, nhead={nhead}, num_layers={num_encoder_layers}, dim_ff={dim_feedforward}")

    # --- B. 設置模型和數據 (!!! 已修改：分割驗證集 !!!) ---
    
    # 1. 加載「完整」的訓練數據集
    #    (shuffle=False 確保每次 HPO 的分割都一致)
    _, full_train_dataset = create_dataloader(FOLDER, "post_vol_", "train", batch_size=BATCH_SIZE, shuffle=False, compute_stats=False)

    # 2. 將「訓練集」分割為「新訓練集」(80%) 和「驗證集」(20%)
    train_indices, val_indices = train_test_split(
        range(len(full_train_dataset)), 
        test_size=0.2, 
        random_state=42 # 固定 random_state 確保可重現性
    )
    
    train_subset = Subset(full_train_dataset, train_indices)
    val_subset = Subset(full_train_dataset, val_indices)
    
    # 3. 創建新的 Dataloader
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False) # 驗證集不需要 shuffle

    # 4. 獲取形狀 (從新的 train_loader)
    first_batch, _, _ = next(iter(train_loader))
    seq_len = first_batch.shape[2] # 41
    input_dim = first_batch.shape[3] # 20
    
    # 5. 根據建議的超參數建立模型 (與之前相同)
    model = TransformerAutoencoder(
        input_dim=input_dim,
        seq_len=seq_len,
        latent_dim=10, # 我們固定 latent_dim=10
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward
    ).to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=N_EPOCHS_PER_TRIAL, eta_min=lr * 0.01)

    best_validation_loss = float('inf') # (!!! 已修改 !!!)

    # --- C. 訓練迴圈 (!!! 已修改：使用驗證集 !!!) ---
    for epoch in range(N_EPOCHS_PER_TRIAL):
        model.train()
        total_train_loss = 0.0
        # (訓練迴圈 ... 保持不變)
        for x, _, _ in train_loader:
            x = x.to(DEVICE)
            optimizer.zero_grad()
            x_recon = model(x)
            loss = F.mse_loss(x_recon, x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_train_loss += loss.item() * x.size(0)
        
        scheduler.step()
        # avg_train_loss = total_train_loss / len(train_loader.dataset) # 應為 len(train_subset)
        
        # 評估「驗證集」 (!!! 已修改 !!!)
        model.eval()
        total_validation_loss = 0.0
        with torch.no_grad():
            for x_val, _, _ in validation_loader: # (!!! 已修改 !!!)
                x_val = x_val.to(DEVICE)
                x_recon_val = model(x_val)
                loss_val = F.mse_loss(x_recon_val, x_val)
                total_validation_loss += loss_val.item() * x_val.size(0)
        
        avg_validation_loss = total_validation_loss / len(val_subset) # (!!! 已修改 !!!)
        
        if avg_validation_loss < best_validation_loss:
            best_validation_loss = avg_validation_loss
        
        # (!!! 已修改 !!!)
        print(f"Trial {trial.number}, Epoch {epoch+1}/{N_EPOCHS_PER_TRIAL} | val_loss={avg_validation_loss:.6f}")

        # --- D. 關鍵的「剪枝」步驟 (!!! 已修改 !!!) ---
        
        # 1. 向 Optuna 報告目前 epoch 的 *validation_loss*
        trial.report(avg_validation_loss, epoch) # (!!! 已修改 !!!)
        
        # 2. 檢查 Optuna 是否認為这个 trial 已經沒希望了
        if trial.should_prune():
            print(f"Pruning trial {trial.number} at epoch {epoch+1} due to poor performance.")
            raise TrialPruned()

    # 迴圈結束後，回傳此 trial 達到的「最佳 validation_loss」
    return best_validation_loss # (!!! 已修改 !!!)

# -------------------------------------------------------------------
# 4. 主執行函數
# -------------------------------------------------------------------
def run_hpo():
    print(f"開始 Optuna HPO ({N_TRIALS} trials, {N_EPOCHS_PER_TRIAL} epochs/trial)...")
    print(f"將在 {DEVICE} 上運行")
    
    # 設置一個「儲存庫」，Optuna 會將結果保存在一個 .db 檔案中
    # 這樣就算程式中斷，也可以從上次的地方繼續
    storage_name = "sqlite:///tae_hpo.db"
    
    # 1. 創建 Study (研究)
    # TPE (Tree-structured Parzen Estimator) 是 Optuna 預設的貝葉斯優化算法
    # Pruner 會自動剪掉「看起來沒希望」的 trial
    study = optuna.create_study(
        study_name="tae-hpo-v1",
        storage=storage_name,
        load_if_exists=True, # 如果 .db 檔案存在，就加載並繼續
        direction="minimize",  # 我們的目標是「最小化」test_loss
        pruner=optuna.pruners.MedianPruner() # 使用「中位數剪枝器」
    )
    
    # 2. 開始優化！
    # 這會運行 N_TRIALS 次 objective 函數
    try:
        study.optimize(objective, n_trials=N_TRIALS)
    except KeyboardInterrupt:
        print("使用者手動停止 HPO。")
    
    # 3. 打印最佳結果
    print("\n--- [HPO 完成] ---")
    print(f"總共完成的 Trial 數量: {len(study.trials)}")
    
    best_trial = study.best_trial
    print("\n🎉 最佳 Trial 找到了 🎉")
    print(f"  Trial 編號: {best_trial.number}")
    print(f"  最佳 Test Loss (MSE): {best_trial.value:.8f}")
    
    print("\n  最佳超參數 (Hyperparameters):")
    for key, value in best_trial.params.items():
        print(f"    - {key}: {value}")

    print("\n--- 如何使用這些參數 ---")
    print("1. 打開 main_Transformer.py")
    print(f"2. 修改 train_and_save_TAE 函數的 lr={best_trial.params['lr']:.2e}")
    print(f"3. 修改 TransformerAutoencoder 的 __init__ 默認值:")
    print(f"     d_model={best_trial.params['d_model']}")
    print(f"     nhead={best_trial.params['nhead']}")
    print(f"     num_encoder_layers={best_trial.params['num_encoder_layers']}")
    print(f"     num_decoder_layers={best_trial.params['num_encoder_layers']}")
    print(f"     dim_feedforward={best_trial.params['dim_feedforward']}")
    print("4. 重新運行 main_Transformer.py (if 1) 來訓練最終的最佳模型。")
    
    print("\n(可選) 運行視覺化儀表板:")
    print(f"optuna-dashboard {storage_name}")


if __name__ == "__main__":
    run_hpo()