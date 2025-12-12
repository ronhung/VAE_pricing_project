import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# 引用您的模型定義
from Transformer_model import PricerTransformer, create_pricing_dataloader
from VAE_model import Pricer as VAEPricer

def run_ablation_study(
    folder="../data_process/data_pack",
    product_type="AsianPut",
    latent_dim=10,
    batch_size=256
):
    print(f"--- Starting Ablation Study for {product_type} ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 準備路徑
    vae_pricer_path = f"{folder}/{product_type}_pricer_state_dict.pt"
    tae_pricer_path = f"{folder}/{product_type}_pricer_tae_state_dict.pt"
    tae_model_path = f"{folder}/tae_model_{latent_dim}d.pt"  # PricerTransformer 初始化需要
    
    # 檢查模型是否存在
    if not os.path.exists(vae_pricer_path) or not os.path.exists(tae_pricer_path):
        print(f"Error: Model files not found. Please ensure training is complete.")
        print(f"Looking for: {vae_pricer_path}")
        print(f"Looking for: {tae_pricer_path}")
        return

    # 2. 加載數據 (使用 Transformer 的 DataLoader，因為它包含 Spot Price S0)
    # 我們需要 S0 來計算 Moneyness
    print("Loading Test Data...")
    test_loader, dataset = create_pricing_dataloader(
        folder, product_type, "test", batch_size=batch_size, shuffle=False, compute_stats=False
    )
    
    # 獲取 Denormalization 統計數據
    param_stats = np.load(f"{folder}/{product_type}_pricing_param_stats.npz")
    price_mean = param_stats["price_mean"]
    price_std = param_stats["price_std"]
    params_mean = param_stats["params_mean"]
    params_std = param_stats["params_std"]
    
    # 兼容舊版數據：檢查是否有 spot_mean/std
    if "spot_mean" in param_stats:
        spot_mean = param_stats["spot_mean"]
        spot_std = param_stats["spot_std"]
    else:
        # 如果檔案中沒有，嘗試從數據集屬性獲取 (如果是剛計算完)
        spot_mean = dataset.spot_mean
        spot_std = dataset.spot_std

    # 3. 初始化模型
    print("Initializing Models...")
    
    # VAE Pricer (Baseline)
    # 獲取輸入形狀
    sample_batch = next(iter(test_loader))
    vol_shape = sample_batch[0].shape[2:] # (H, W)
    
    vae_model = VAEPricer(
        latent_dim=latent_dim, 
        pricing_param_dim=2, # K, T
        vol_input_shape=vol_shape
    ).to(device)
    vae_model.load_state_dict(torch.load(vae_pricer_path, map_location=device))
    vae_model.eval()
    
    # Transformer Pricer (Ours)
    tae_model = PricerTransformer(
        latent_dim=latent_dim,
        pricing_param_dim=2,
        spot_param_dim=1, # 包含 S0
        tae_model_path=tae_model_path # 初始化需要，權重會被下面的 load_state_dict 覆蓋
    ).to(device)
    tae_model.load_state_dict(torch.load(tae_pricer_path, map_location=device))
    tae_model.eval()
    
    # 4. 批量推理
    results = []
    
    print("Running Inference...")
    with torch.no_grad():
        for vol, params, spot, target in test_loader:
            vol = vol.to(device)
            params = params.to(device)
            spot = spot.to(device)
            target = target.to(device)
            
            # VAE Inference (只接受 vol 和 params)
            vae_pred = vae_model(vol, params)
            
            # Transformer Inference (接受 vol, params, spot)
            tae_pred = tae_model(vol, params, spot)
            
            # 收集數據 (轉回 CPU numpy)
            batch_res = {
                "vae_pred": vae_pred.cpu().numpy().flatten(),
                "tae_pred": tae_pred.cpu().numpy().flatten(),
                "target": target.cpu().numpy().flatten(),
                "params": params.cpu().numpy(),
                "spot": spot.cpu().numpy().flatten()
            }
            results.append(batch_res)
            
    # 5. 數據整合與還原 (Denormalization)
    print("Processing Results...")
    vae_preds = np.concatenate([r["vae_pred"] for r in results])
    tae_preds = np.concatenate([r["tae_pred"] for r in results])
    targets = np.concatenate([r["target"] for r in results])
    params_norm = np.concatenate([r["params"] for r in results])
    spots_norm = np.concatenate([r["spot"] for r in results])
    
    # 還原數值
    real_vae_price = vae_preds * price_std + price_mean
    real_tae_price = tae_preds * price_std + price_mean
    real_target_price = targets * price_std + price_mean
    
    real_params = params_norm * params_std + params_mean
    real_K = real_params[:, 0]
    real_T = real_params[:, 1]
    
    real_S0 = spots_norm * spot_std + spot_mean
    
    # 計算 Moneyness (M = S/K for Put, or K/S usually. Let's use Log Moneyness k = log(K/S))
    # 注意：您的數據集中 K 是 Strike。 Log Moneyness 通常定義為 ln(K/S) 或 ln(S/K)
    # 這裡我們使用 ln(S/K) (對於 Put, 越小越 Deep OTM? 或是反過來)
    # 簡單起見：使用 Moneyness Ratio M = S / K
    # 對於 Put: 
    #   S/K << 1 (S < K) => ITM
    #   S/K >> 1 (S > K) => OTM
    # 讓我們用標準的 Log-Moneyness from model input usually: k = log(K/S)
    # 但這裡我們直接算:
    moneyness = np.log(real_K / real_S0) 
    
    # 建立 DataFrame
    df = pd.DataFrame({
        "Strike": real_K,
        "Maturity": real_T,
        "Spot": real_S0,
        "LogMoneyness": moneyness, # >0 -> K>S (ITM for Put), <0 -> K<S (OTM for Put)
        "TargetPrice": real_target_price,
        "VAE_Price": real_vae_price,
        "TAE_Price": real_tae_price,
        "VAE_Error": np.abs(real_vae_price - real_target_price),
        "TAE_Error": np.abs(real_tae_price - real_target_price)
    })
    
    # 6. 分組分析 (Bucketing)
    # 定義區間
    # LogMoneyness buckets: Deep OTM, OTM, ATM, ITM, Deep ITM
    # For Put:
    #   Log(K/S) < -0.1 : Deep OTM (K much smaller than S)
    #   -0.1 ~ -0.02    : OTM
    #   -0.02 ~ 0.02    : ATM
    #   0.02 ~ 0.1      : ITM
    #   > 0.1           : Deep ITM
    
    m_bins = [-np.inf, -0.15, -0.05, 0.05, 0.15, np.inf]
    m_labels = ["Deep OTM", "OTM", "ATM", "ITM", "Deep ITM"]
    
    t_bins = [0, 0.25, 0.5, 0.75, 1.5]
    t_labels = ["Short (<3m)", "Medium (3-6m)", "Long (6-9m)", "Very Long (>9m)"]
    
    df["Moneyness_Bin"] = pd.cut(df["LogMoneyness"], bins=m_bins, labels=m_labels)
    df["Maturity_Bin"] = pd.cut(df["Maturity"], bins=t_bins, labels=t_labels)
    
    # 計算每個 Bin 的 MAE
    heatmap_vae = df.pivot_table(index="Moneyness_Bin", columns="Maturity_Bin", values="VAE_Error", aggfunc="mean")
    heatmap_tae = df.pivot_table(index="Moneyness_Bin", columns="Maturity_Bin", values="TAE_Error", aggfunc="mean")
    
    # 計算改進百分比 (VAE - TAE) / VAE
    heatmap_imp = (heatmap_vae - heatmap_tae) / heatmap_vae * 100
    
    # 7. 繪圖
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    sns.heatmap(heatmap_vae, annot=True, fmt=".4f", cmap="Reds", ax=axes[0])
    axes[0].set_title(f"Baseline (VAE) MAE Error\n({product_type})")
    
    sns.heatmap(heatmap_tae, annot=True, fmt=".4f", cmap="Reds", ax=axes[1])
    axes[1].set_title(f"Ours (Transformer) MAE Error\n({product_type})")
    
    sns.heatmap(heatmap_imp, annot=True, fmt=".1f", cmap="RdYlGn", center=0, ax=axes[2])
    axes[2].set_title(f"Improvement % (Green = Better)\n(Positive means TAE error is lower)")
    
    plt.tight_layout()
    save_path = f"{folder}/ablation_study_{product_type}.png"
    plt.savefig(save_path)
    print(f"Comparison Heatmap saved to {save_path}")
    plt.show()
    
    # 8. 輸出關鍵統計
    print("\n====== Ablation Summary ======")
    print(f"Overall VAE MAE: {df['VAE_Error'].mean():.4f}")
    print(f"Overall TAE MAE: {df['TAE_Error'].mean():.4f}")
    
    # 找出改善最多的區域
    print("\nPerformance in Deep OTM & Long Maturity:")
    try:
        dotm_long_vae = heatmap_vae.loc["Deep OTM", "Very Long (>9m)"]
        dotm_long_tae = heatmap_tae.loc["Deep OTM", "Very Long (>9m)"]
        print(f"  Deep OTM / Long Mat - VAE MAE: {dotm_long_vae:.4f}")
        print(f"  Deep OTM / Long Mat - TAE MAE: {dotm_long_tae:.4f}")
        print(f"  Improvement: {(dotm_long_vae - dotm_long_tae)/dotm_long_vae*100:.2f}%")
    except:
        print("  (Specific bin not present in test data)")

if __name__ == "__main__":
    # 您可以修改這裡的參數來跑不同的產品
    run_ablation_study(product_type="AsianPut", latent_dim=10)
    # run_ablation_study(product_type="AsianCall", latent_dim=10)