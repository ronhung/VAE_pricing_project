# -------------------------------------------------------------------
# main_PCA.py
# 
# 仿造 main_VAE.py，但使用 PCA 進行降維
# -------------------------------------------------------------------

from PCA_model import *
from torch.utils.data import Subset


def main():
    folder = "../data_process/data_pack"
    ld = 10  # 潛在維度 (主成分數量)
    pca_model_path = f"{folder}/pca_model_{ld}d.joblib"
    
    if 1:
        # 步驟 1: "訓練" PCA 模型 (擬合數據並保存)
        print(f"--- 訓練 PCA 模型 (n_components={ld}) ---")
        train_and_save_PCA(folder, latent_dim=ld, save_path=pca_model_path)
        print("--- PCA 模型訓練完成 ---")

    if 1:
        # 步驟 2: 訓練 AmericanPut 定價器
        print("--- 訓練 AmericanPut Pricer (PCA) ---")
        pricer_path = f"{folder}/AmericanPut_pricer_pca_state_dict.pt"
        train_and_save_pricer_PCA(
            folder, 
            product_type="AmericanPut", 
            pca_model_path=pca_model_path,  # 使用 .joblib
            latent_dim=ld, 
            pricing_param_dim=2, 
            num_epochs=150  # 只有一個訓練階段
        )
        
        plot_predict_prices_from_vol_surface_and_params(
            folder=folder,
            product_type="AmericanPut",
            pricer_model_path=pricer_path,
            pca_model_path=pca_model_path, # 傳遞 PCA 路徑給 PricerPCA
            include_train=True,
            latent_dim=ld,
            pricing_param_dim=2
        )
        plot_loss_curves(folder, product_type="AmericanPut", model_type="PCA")
        print("--- AmericanPut Pricer (PCA) 訓練完成 ---")

    if 1:
        # 步驟 3: 訓練 AsianCall 定價器
        print("--- 訓練 AsianCall Pricer (PCA) ---")
        pricer_path = f"{folder}/AsianCall_pricer_pca_state_dict.pt"
        train_and_save_pricer_PCA(
            folder, 
            product_type="AsianCall", 
            pca_model_path=pca_model_path, 
            latent_dim=ld, 
            pricing_param_dim=2, 
            num_epochs=150
        )
        
        plot_predict_prices_from_vol_surface_and_params(
            folder=folder,
            product_type="AsianCall",
            pricer_model_path=pricer_path,
            pca_model_path=pca_model_path,
            include_train=True,
            latent_dim=ld,
            pricing_param_dim=2
        )
        plot_loss_curves(folder, product_type="AsianCall", model_type="PCA")
        print("--- AsianCall Pricer (PCA) 訓練完成 ---")

    if 1:
        # 步驟 4: 訓練 AsianPut 定價器
        print("--- 訓練 AsianPut Pricer (PCA) ---")
        pricer_path = f"{folder}/AsianPut_pricer_pca_state_dict.pt"
        train_and_save_pricer_PCA(
            folder, 
            product_type="AsianPut", 
            pca_model_path=pca_model_path, 
            latent_dim=ld, 
            pricing_param_dim=2, 
            num_epochs=150
        )
        
        plot_predict_prices_from_vol_surface_and_params(
            folder=folder,
            product_type="AsianPut",
            pricer_model_path=pricer_path,
            pca_model_path=pca_model_path,
            include_train=True,
            latent_dim=ld,
            pricing_param_dim=2
        )
        plot_loss_curves(folder, product_type="AsianPut", model_type="PCA")
        print("--- AsianPut Pricer (PCA) 訓練完成 ---")

    # 步驟 5: 可視化
    visualize_latent_distribution_PCA(
        pca_model_path, 
        folder, 
        save_path=f"{folder}/pca_latent_distribution.png"
    )

    if 0: # 可選：顯示重建效果
        quote_dates = ["2020-03-10", "2021-06-15", "2023-11-29"]
        show_quote_date_reconstructions_PCA(
            folder=folder,
            quote_dates=quote_dates,
            model_path=pca_model_path,
            latent_dim=ld
        )


if __name__ == "__main__":
    main()