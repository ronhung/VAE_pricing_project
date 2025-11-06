# -------------------------------------------------------------------
# main_Transformer.py
# 
# 仿造 main_PCA.py，但使用 TransformerAutoencoder (TAE) 進行降維
# -------------------------------------------------------------------

# CHANGED: 導入 Transformer_model
from Transformer_model import *
from torch.utils.data import Subset


def main():
    folder = "../data_process/data_pack"
    ld = 10  # 潛在維度
    
    # CHANGED: 模型路徑 (PyTorch state_dict)
    tae_model_path = f"{folder}/tae_model_{ld}d.pt"
    
    if 1:
        # 步驟 1: 訓練 TransformerAutoencoder (TAE) 模型
        print(f"--- 訓練 TAE 模型 (latent_dim={ld}) ---")
        # CHANGED:
        train_and_save_TAE(
            folder, 
            latent_dim=ld,
            num_epochs=100, # TAE 訓練 100 epochs
            batch_size=64, # Transformer 偏好稍大的 batch
            lr=9.59e-04,
            save_path=tae_model_path
        )
        plot_tae_loss_curves(folder) # 繪製 TAE 自身的損失曲線
        print("--- TAE 模型訓練完成 ---")

    if 1:
        # 步驟 2: 訓練 AmericanPut 定價器
        print("--- 訓練 AmericanPut Pricer (TAE) ---")
        # CHANGED:
        pricer_path = f"{folder}/AmericanPut_pricer_tae_state_dict.pt"
        train_and_save_pricer_TAE(
            folder, 
            product_type="AmericanPut", 
            tae_model_path=tae_model_path,
            latent_dim=ld, 
            pricing_param_dim=2, 
            num_epochs=150
        )
        
        # CHANGED:
        plot_predict_prices_from_vol_surface_and_params_TAE(
            folder=folder,
            product_type="AmericanPut",
            pricer_model_path=pricer_path,
            tae_model_path=tae_model_path,
            include_train=True,
            latent_dim=ld,
            pricing_param_dim=2
        )
        # CHANGED:
        plot_loss_curves(folder, product_type="AmericanPut", model_type="TAE")
        print("--- AmericanPut Pricer (TAE) 訓練完成 ---")

    if 1:
        # 步驟 3: 訓練 AsianCall 定價器
        print("--- 訓練 AsianCall Pricer (TAE) ---")
        # CHANGED:
        pricer_path = f"{folder}/AsianCall_pricer_tae_state_dict.pt"
        train_and_save_pricer_TAE(
            folder, 
            product_type="AsianCall", 
            tae_model_path=tae_model_path, 
            latent_dim=ld, 
            pricing_param_dim=2, 
            num_epochs=150
        )
        
        # CHANGED:
        plot_predict_prices_from_vol_surface_and_params_TAE(
            folder=folder,
            product_type="AsianCall",
            pricer_model_path=pricer_path,
            tae_model_path=tae_model_path,
            include_train=True,
            latent_dim=ld,
            pricing_param_dim=2
        )
        # CHANGED:
        plot_loss_curves(folder, product_type="AsianCall", model_type="TAE")
        print("--- AsianCall Pricer (TAE) 訓練完成 ---")

    if 1:
        # 步驟 4: 訓練 AsianPut 定價器
        print("--- 訓練 AsianPut Pricer (TAE) ---")
        # CHANGED:
        pricer_path = f"{folder}/AsianPut_pricer_tae_state_dict.pt"
        train_and_save_pricer_TAE(
            folder, 
            product_type="AsianPut", 
            tae_model_path=tae_model_path, 
            latent_dim=ld, 
            pricing_param_dim=2, 
            num_epochs=150
        )
        
        # CHANGED:
        plot_predict_prices_from_vol_surface_and_params_TAE(
            folder=folder,
            product_type="AsianPut",
            pricer_model_path=pricer_path,
            tae_model_path=tae_model_path,
            include_train=True,
            latent_dim=ld,
            pricing_param_dim=2
        )
        # CHANGED:
        plot_loss_curves(folder, product_type="AsianPut", model_type="TAE")
        print("--- AsianPut Pricer (TAE) 訓練完成 ---")

    # 步驟 5: 可視化潛在空間
    # CHANGED:
    visualize_latent_distribution_TAE(
        tae_model_path, 
        folder, 
        latent_dim=ld,
        save_path=f"{folder}/tae_latent_distribution.png"
    )

    if 1: # TAE 可以重建!
        quote_dates = ["2020-03-10", "2021-06-15", "2023-11-29"]
        # CHANGED:
        show_quote_date_reconstructions_TAE(
            folder=folder,
            quote_dates=quote_dates,
            model_path=tae_model_path,
            latent_dim=ld
        )


if __name__ == "__main__":
    main()