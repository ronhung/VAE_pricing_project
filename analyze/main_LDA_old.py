# -------------------------------------------------------------------
# main_LDA.py
# 
# 仿造 main_PCA.py，但使用 LDA 進行降維
# -------------------------------------------------------------------

# CHANGED: 導入 LDA_model
from LDA_model import *
from torch.utils.data import Subset


def main():
    folder = "../data_process/data_pack"
    ld = 10  # 潛在維度 (LDA components)
    k_clusters = 11 # K-Means 群集數 (必須 > ld)
    
    # CHANGED: 模型路徑
    lda_model_path = f"{folder}/lda_kmeans_model_{ld}d_{k_clusters}k.joblib"
    
    if 1:
        # 步驟 1: "訓練" LDA/K-Means 模型 (擬合數據並保存)
        print(f"--- 訓練 LDA/K-Means 模型 (n_components={ld}, n_clusters={k_clusters}) ---")
        # CHANGED: 
        train_and_save_LDA(
            folder, 
            latent_dim=ld, 
            n_clusters=k_clusters, 
            save_path=lda_model_path
        )
        print("--- LDA/K-Means 模型訓練完成 ---")

    if 1:
        # 步驟 2: 訓練 AmericanPut 定價器
        print("--- 訓練 AmericanPut Pricer (LDA) ---")
        # CHANGED:
        pricer_path = f"{folder}/AmericanPut_pricer_lda_state_dict.pt"
        train_and_save_pricer_LDA(
            folder, 
            product_type="AmericanPut", 
            lda_model_path=lda_model_path,  # 使用 .joblib
            latent_dim=ld, 
            pricing_param_dim=2, 
            num_epochs=150
        )
        
        # CHANGED:
        plot_predict_prices_from_vol_surface_and_params_LDA(
            folder=folder,
            product_type="AmericanPut",
            pricer_model_path=pricer_path,
            lda_model_path=lda_model_path, # 傳遞 LDA 路徑
            include_train=True,
            latent_dim=ld,
            pricing_param_dim=2
        )
        # CHANGED:
        plot_loss_curves(folder, product_type="AmericanPut", model_type="LDA")
        print("--- AmericanPut Pricer (LDA) 訓練完成 ---")

    if 1:
        # 步驟 3: 訓練 AsianCall 定價器
        print("--- 訓練 AsianCall Pricer (LDA) ---")
        # CHANGED:
        pricer_path = f"{folder}/AsianCall_pricer_lda_state_dict.pt"
        train_and_save_pricer_LDA(
            folder, 
            product_type="AsianCall", 
            lda_model_path=lda_model_path, 
            latent_dim=ld, 
            pricing_param_dim=2, 
            num_epochs=150
        )
        
        # CHANGED:
        plot_predict_prices_from_vol_surface_and_params_LDA(
            folder=folder,
            product_type="AsianCall",
            pricer_model_path=pricer_path,
            lda_model_path=lda_model_path,
            include_train=True,
            latent_dim=ld,
            pricing_param_dim=2
        )
        # CHANGED:
        plot_loss_curves(folder, product_type="AsianCall", model_type="LDA")
        print("--- AsianCall Pricer (LDA) 訓練完成 ---")

    if 1:
        # 步驟 4: 訓練 AsianPut 定價器
        print("--- 訓練 AsianPut Pricer (LDA) ---")
        # CHANGED:
        pricer_path = f"{folder}/AsianPut_pricer_lda_state_dict.pt"
        train_and_save_pricer_LDA(
            folder, 
            product_type="AsianPut", 
            lda_model_path=lda_model_path, 
            latent_dim=ld, 
            pricing_param_dim=2, 
            num_epochs=150
        )
        
        # CHANGED:
        plot_predict_prices_from_vol_surface_and_params_LDA(
            folder=folder,
            product_type="AsianPut",
            pricer_model_path=pricer_path,
            lda_model_path=lda_model_path,
            include_train=True,
            latent_dim=ld,
            pricing_param_dim=2
        )
        # CHANGED:
        plot_loss_curves(folder, product_type="AsianPut", model_type="LDA")
        print("--- AsianPut Pricer (LDA) 訓練完成 ---")

    # 步驟 5: 可視化
    # CHANGED:
    visualize_latent_distribution_LDA(
        lda_model_path, 
        folder, 
        save_path=f"{folder}/lda_latent_distribution.png"
    )

    if 0: 
        # NOTE: 此功能無法實現
        # LDA (LinearDiscriminantAnalysis) 沒有 `inverse_transform` 方法。
        # 它不像 PCA 那樣是可逆的，因為它會丟棄與類別分離無關的變異數。
        # 因此，我們無法重建波動率曲面。
        pass
        # show_quote_date_reconstructions_LDA(...) 


if __name__ == "__main__":
    main()