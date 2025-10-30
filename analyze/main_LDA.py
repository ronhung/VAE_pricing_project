# -------------------------------------------------------------------
# main_LDA.py
# 
# 仿造 main_PCA.py，但使用 LDA (v2, price-based labels)
# -------------------------------------------------------------------

from LDA_model import *
from torch.utils.data import Subset


def main():
    folder = "../data_process/data_pack"
    ld = 10  # 期望的潛在維度 (LDA components)
    n_bins = 11 # 價格分箱的類別數 (必須 > ld)
    
    # 由於 LDA 模型現在是 product-specific，
    # 我們將在每個 product block 中單獨訓練和加載
    
    # -----------------------------------------------------
    # 步驟 1: AmericanPut
    # -----------------------------------------------------
    if 1:
        product = "AmericanPut"
        print(f"--- [START] {product} (LDA v2) ---")
        
        # 1a: 訓練 product-specific LDA model
        lda_model_path = f"{folder}/{product}_lda_price_labels_model.joblib"
        pricer_path = f"{folder}/{product}_pricer_lda_state_dict.pt"
        
        print(f"--- 訓練 {product} LDA 模型 (n_components={ld}, n_bins={n_bins}) ---")
        train_and_save_LDA(
            folder,
            product_type=product,
            latent_dim=ld,
            n_bins=n_bins,
            save_path=lda_model_path
        )
        print(f"--- {product} LDA 模型訓練完成 ---")

        # 1b: 訓練 Pricer
        print(f"--- 訓練 {product} Pricer (LDA) ---")
        train_and_save_pricer_LDA(
            folder, 
            product_type=product, 
            lda_model_path=lda_model_path,
            latent_dim=ld, 
            pricing_param_dim=2, 
            num_epochs=150,
            weight_decay=1e-3 # 使用較高的 L2
        )
        
        # 1c: 繪圖
        plot_predict_prices_from_vol_surface_and_params_LDA(
            folder=folder,
            product_type=product,
            pricer_model_path=pricer_path,
            lda_model_path=lda_model_path,
            include_train=True,
            latent_dim=ld,
            pricing_param_dim=2
        )
        plot_loss_curves(folder, product_type=product, model_type="LDA")
        
        # 1d: 可視化潛在空間
        visualize_latent_distribution_LDA(
            lda_model_path, 
            folder,
            product_type=product, # 指定產品
            save_path=f"{folder}/{product}_lda_latent_distribution.png"
        )
        print(f"--- [END] {product} (LDA v2) ---")

    # -----------------------------------------------------
    # 步驟 2: AsianCall
    # -----------------------------------------------------
    if 1:
        product = "AsianCall"
        print(f"--- [START] {product} (LDA v2) ---")
        
        # 2a: 訓練 product-specific LDA model
        lda_model_path = f"{folder}/{product}_lda_price_labels_model.joblib"
        pricer_path = f"{folder}/{product}_pricer_lda_state_dict.pt"
        
        print(f"--- 訓練 {product} LDA 模型 (n_components={ld}, n_bins={n_bins}) ---")
        train_and_save_LDA(
            folder,
            product_type=product,
            latent_dim=ld,
            n_bins=n_bins,
            save_path=lda_model_path
        )
        print(f"--- {product} LDA 模型訓練完成 ---")

        # 2b: 訓練 Pricer
        print(f"--- 訓練 {product} Pricer (LDA) ---")
        train_and_save_pricer_LDA(
            folder, 
            product_type=product, 
            lda_model_path=lda_model_path,
            latent_dim=ld, 
            pricing_param_dim=2, 
            num_epochs=150,
            weight_decay=1e-3
        )
        
        # 2c: 繪圖
        plot_predict_prices_from_vol_surface_and_params_LDA(
            folder=folder,
            product_type=product,
            pricer_model_path=pricer_path,
            lda_model_path=lda_model_path,
            include_train=True,
            latent_dim=ld,
            pricing_param_dim=2
        )
        plot_loss_curves(folder, product_type=product, model_type="LDA")
        
        # 2d: 可視化潛在空間
        visualize_latent_distribution_LDA(
            lda_model_path, 
            folder,
            product_type=product,
            save_path=f"{folder}/{product}_lda_latent_distribution.png"
        )
        print(f"--- [END] {product} (LDA v2) ---")

    # -----------------------------------------------------
    # 步驟 3: AsianPut
    # -----------------------------------------------------
    if 1:
        product = "AsianPut"
        print(f"--- [START] {product} (LDA v2) ---")
        
        # 3a: 訓練 product-specific LDA model
        lda_model_path = f"{folder}/{product}_lda_price_labels_model.joblib"
        pricer_path = f"{folder}/{product}_pricer_lda_state_dict.pt"
        
        print(f"--- 訓練 {product} LDA 模型 (n_components={ld}, n_bins={n_bins}) ---")
        train_and_save_LDA(
            folder,
            product_type=product,
            latent_dim=ld,
            n_bins=n_bins,
            save_path=lda_model_path
        )
        print(f"--- {product} LDA 模型訓練完成 ---")

        # 3b: 訓練 Pricer
        print(f"--- 訓練 {product} Pricer (LDA) ---")
        train_and_save_pricer_LDA(
            folder, 
            product_type=product, 
            lda_model_path=lda_model_path,
            latent_dim=ld, 
            pricing_param_dim=2, 
            num_epochs=150,
            weight_decay=1e-3
        )
        
        # 3c: 繪圖
        plot_predict_prices_from_vol_surface_and_params_LDA(
            folder=folder,
            product_type=product,
            pricer_model_path=pricer_path,
            lda_model_path=lda_model_path,
            include_train=True,
            latent_dim=ld,
            pricing_param_dim=2
        )
        plot_loss_curves(folder, product_type=product, model_type="LDA")
        
        # 3d: 可視化潛在空間
        visualize_latent_distribution_LDA(
            lda_model_path, 
            folder,
            product_type=product,
            save_path=f"{folder}/{product}_lda_latent_distribution.png"
        )
        print(f"--- [END] {product} (LDA v2) ---")


if __name__ == "__main__":
    main()