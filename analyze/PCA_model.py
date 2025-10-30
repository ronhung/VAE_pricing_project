# -------------------------------------------------------------------
# PCA_model.py
#
# 仿造 VAE_model.py 的結構，但使用 sklearn.decomposition.PCA
# 替換 VAE 的 Encoder 和 Decoder。
# Pricer 模型將包含一個 *固定* (非訓練) 的 PCA 轉換層。
# -------------------------------------------------------------------

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ML_analyze import * # 假設 ML_analyze.py 存在
from sklearn.decomposition import PCA
import joblib # 用於保存/加載 sklearn 模型
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Optional, Union

# --------------------------
# Data Loading Functions
# (與 VAE_model.py 完全相同)
# --------------------------

class VolsurfaceDataset(Dataset):
    def __init__(self, folder, label, data_type, compute_stats=False):
        # start with simple parsing
        # get raw vol data
        vol_data_path = f"{folder}/{label}grid_{data_type}.npz"
        vol_data = np.load(vol_data_path)

        self.k_grid = vol_data["k_grid"]
        self.T_grid = vol_data["T_grid"]
        self.quote_dates = vol_data["quote_dates"]
        self.surfaces_grid = vol_data["surfaces_grid"]

        # Normalization statistics file path
        self.stats_path = f"{folder}/vol_normalization_stats.npz"

        if compute_stats:
            # Compute normalization statistics from the data
            self.norm_mean = np.mean(self.surfaces_grid)
            self.norm_std = np.std(self.surfaces_grid)
            # Save to file
            np.savez(self.stats_path, mean=self.norm_mean, std=self.norm_std)
            print(f"Computed normalization stats: mean={self.norm_mean:.6f}, std={self.norm_std:.6f}")
            print(f"Saved normalization stats to {self.stats_path}")
        else:
            # Load existing normalization statistics
            if os.path.exists(self.stats_path):
                stats = np.load(self.stats_path)
                self.norm_mean = stats["mean"].item()
                self.norm_std = stats["std"].item()
                print(f"Loaded normalization stats: mean={self.norm_mean:.6f}, std={self.norm_std:.6f}")
            else:
                # If no stats file exists, compute them (fallback)
                print(f"Warning: No normalization stats found at {self.stats_path}, computing from current data")
                self.norm_mean = np.mean(self.surfaces_grid)
                self.norm_std = np.std(self.surfaces_grid)
                np.savez(self.stats_path, mean=self.norm_mean, std=self.norm_std)

        # Normalize the data
        self.surfaces_grid = (self.surfaces_grid - self.norm_mean) / self.norm_std

    def __len__(self):
        return len(self.quote_dates)

    def __getitem__(self, idx):
        surface_grid = self.surfaces_grid[idx]
        surface_grid = torch.from_numpy(surface_grid).float().unsqueeze(0)  # add channel
        k_grid = torch.tensor(self.k_grid, dtype=torch.float32)
        T_grid = torch.tensor(self.T_grid, dtype=torch.float32)
        return surface_grid, k_grid, T_grid


def create_dataloader(folder, label, data_type, batch_size=32, shuffle=True, compute_stats=False):
    dataset = VolsurfaceDataset(folder, label, data_type, compute_stats=compute_stats)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle), dataset


class PricingDataset(Dataset):
    def __init__(self, folder, product_type, data_type, compute_param_stats=False):
        """
        Dataset for pricing data that contains vol surfaces and corresponding option prices.
        """
        # Load pricing data
        pricing_data_path = f"{folder}/{product_type}_pricing_data_{data_type}.npz"
        print(f"Loading pricing data from {pricing_data_path}")
        pricing_data = np.load(pricing_data_path, allow_pickle=True)

        # Extract data arrays
        if product_type == "AmericanPut" or product_type == "AsianCall" or product_type == "AsianPut":
            self.quote_dates = pricing_data["quote_dates"]
            self.vol_surfaces = pricing_data["vol_surfaces"]
            self.price_params = np.column_stack((pricing_data["K"], pricing_data["T"]))
            self.prices = pricing_data["NPV"]
        else:
            raise ValueError(f"Unsupported product type: {product_type}")

        # Load normalization statistics for vol surfaces
        vol_stats_path = f"{folder}/vol_normalization_stats.npz"
        if os.path.exists(vol_stats_path):
            stats = np.load(vol_stats_path)
            self.norm_mean = stats["mean"].item()
            self.norm_std = stats["std"].item()
        else:
            raise FileNotFoundError(f"Normalization stats not found at {vol_stats_path}")

        # Normalize vol surfaces
        self.vol_surfaces = (self.vol_surfaces - self.norm_mean) / self.norm_std

        # Normalize pricing parameters and prices
        param_stats_path = f"{folder}/{product_type}_pricing_param_stats.npz"
        if compute_param_stats:
            self.price_params_mean = np.mean(self.price_params, axis=0)
            self.price_params_std = np.std(self.price_params, axis=0)
            self.price_mean = np.mean(self.prices)
            self.price_std = np.std(self.prices)
            price_param_stats = {
                "params_mean": self.price_params_mean, "params_std": self.price_params_std,
                "price_mean": self.price_mean, "price_std": self.price_std,
            }
            np.savez(param_stats_path, **price_param_stats)
        else:
            if os.path.exists(param_stats_path):
                stats = np.load(param_stats_path)
                self.price_params_mean = stats["params_mean"]
                self.price_params_std = stats["params_std"]
                self.price_mean = stats["price_mean"]
                self.price_std = stats["price_std"]
            else:
                warnings.warn(f"No pricing parameter stats found at {param_stats_path}, computing from current data")
                # Fallback
                self.price_params_mean = np.mean(self.price_params, axis=0)
                self.price_params_std = np.std(self.price_params, axis=0)
                self.price_mean = np.mean(self.prices)
                self.price_std = np.std(self.prices)


        # Apply normalization to pricing parameters and prices
        self.price_params = (self.price_params - self.price_params_mean) / self.price_params_std
        self.prices = (self.prices - self.price_mean) / self.price_std

    def __len__(self):
        return len(self.quote_dates)

    def __getitem__(self, idx):
        vol_surface = torch.from_numpy(self.vol_surfaces[idx]).float().unsqueeze(0)  # add channel dimension
        pricing_param = torch.tensor(self.price_params[idx], dtype=torch.float32)
        price = torch.tensor(self.prices[idx], dtype=torch.float32).unsqueeze(0)  # make it (1,) for consistency
        return vol_surface, pricing_param, price


def create_pricing_dataloader(folder, label, data_type, batch_size=32, shuffle=True, compute_stats=False):
    """Create a DataLoader for pricing data."""
    dataset = PricingDataset(folder, label, data_type, compute_param_stats=compute_stats)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle), dataset

# --------------------------
# Model Definitions (PCA)
# --------------------------

class PCAWrapper:
    """
    一個包裝 sklearn.decomposition.PCA 的類，
    使其易於擬合、轉換、保存和加載。
    """
    def __init__(self, n_components=10):
        self.pca = PCA(n_components=n_components)
        self.n_components = n_components
        self._is_fitted = False

    def fit(self, data_loader: DataLoader):
        """從 dataloader 擬合 PCA 模型"""
        all_surfaces = []
        print("Gathering all surfaces for PCA fitting...")
        for batch, _, _ in data_loader:
            # batch shape is (batch_size, 1, H, W)
            all_surfaces.append(batch.numpy())
        
        surfaces_np = np.concatenate(all_surfaces, axis=0)
        # 展平為 (num_samples, H*W)
        surfaces_flat = surfaces_np.reshape(surfaces_np.shape[0], -1)
        
        print(f"Fitting PCA on {surfaces_flat.shape[0]} samples with {surfaces_flat.shape[1]} features...")
        self.pca.fit(surfaces_flat)
        self._is_fitted = True
        print("PCA fitting complete.")
        
        # 打印解釋的方差
        explained_variance = self.pca.explained_variance_ratio_ * 100
        total_explained_variance = np.sum(explained_variance)
        print(f"Explained variance per component (%): {explained_variance}")
        print(f"Total explained variance by {self.n_components} components: {total_explained_variance:.2f}%")

    def transform(self, x_batch_tensor: torch.Tensor) -> torch.Tensor:
        """
        將一批波動率曲面 (tensor) 轉換為潛在向量 (tensor)。
        x_batch_tensor: (batch_size, 1, H, W)
        """
        if not self._is_fitted:
            raise RuntimeError("PCA model is not fitted yet. Call .fit() first.")
        
        # 展平並轉為 numpy
        x_flat_np = x_batch_tensor.view(x_batch_tensor.size(0), -1).detach().cpu().numpy()
        
        # 轉換
        z_np = self.pca.transform(x_flat_np)
        
        # 轉回 tensor
        return torch.tensor(z_np, dtype=torch.float32, device=x_batch_tensor.device)

    def inverse_transform(self, z_batch_tensor: torch.Tensor, output_shape: tuple = (1, 41, 20)) -> torch.Tensor:
        """
        將一批潛在向量 (tensor) 逆轉換為波動率曲面 (tensor)。
        z_batch_tensor: (batch_size, latent_dim)
        """
        if not self._is_fitted:
            raise RuntimeError("PCA model is not fitted yet.")
            
        z_np = z_batch_tensor.detach().cpu().numpy()
        
        # 逆轉換
        x_flat_np = self.pca.inverse_transform(z_np)
        
        # 重塑為 (batch_size, 1, H, W)
        x_recon_np = x_flat_np.reshape(z_batch_tensor.size(0), *output_shape)
        
        return torch.tensor(x_recon_np, dtype=torch.float32, device=z_batch_tensor.device)

    def save(self, path: str):
        """使用 joblib 保存擬合好的 PCA 模型"""
        if not self._is_fitted:
            raise RuntimeError("Cannot save an unfitted PCA model.")
        joblib.dump(self.pca, path)
        print(f"PCA model saved to {path}")

    def load(self, path: str):
        """使用 joblib 加載擬合好的 PCA 模型"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"PCA model file not found at {path}")
        self.pca = joblib.load(path)
        self.n_components = self.pca.n_components
        self._is_fitted = True
        print(f"PCA model loaded from {path} (n_components={self.n_components})")


class PricerPCA(nn.Module):
    """
    Pricer 模型，使用 *固定* 的 PCA 進行降維。
    """
    def __init__(self, latent_dim: int, pricing_param_dim: int, pca_model_path: str, vol_input_shape: tuple = (41, 20)):
        super().__init__()
        self.latent_dim = latent_dim
        self.pricing_param_dim = pricing_param_dim
        self.vol_input_shape = vol_input_shape

        # 加載預先擬合的 PCA 模型
        self.pca_wrapper = PCAWrapper(n_components=latent_dim)
        self.pca_wrapper.load(pca_model_path)

        # MLP for pricing: 
        # 輸入維度 = latent_dim (來自 PCA) + pricing_param_dim
        self.pricing_mlp = nn.Sequential(
            nn.Linear(latent_dim + pricing_param_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, vol_surface: torch.Tensor, pricing_params: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vol_surface: (batch_size, 1, H, W) - normalized vol surface
            pricing_params: (batch_size, pricing_param_dim) - pricing parameters

        Returns:
            price_pred: (batch_size, 1) - predicted price
        """
        # 1. 使用 PCA 轉換波動率曲面
        # 這個操作不在 PyTorch 的計算圖中，因為 PCA 是固定的
        # vol_surface shape: (batch_size, 1, H, W)
        with torch.no_grad():
            # 確保 vol_surface 在正確的設備上
            # .transform 內部會處理 .cpu()
            z = self.pca_wrapper.transform(vol_surface) 
            z = z.to(vol_surface.device) # 確保 z 回到 GPU (如果適用)

        # 2. 串聯潛在向量和定價參數
        # z shape: (batch_size, latent_dim)
        # pricing_params shape: (batch_size, pricing_param_dim)
        mlp_input = torch.cat([z, pricing_params], dim=1)  # (batch_size, latent_dim + pricing_param_dim)

        # 3. 通過 MLP 進行定價
        price_pred = self.pricing_mlp(mlp_input)
        
        # 注意：這裡沒有 VAE 的多重採樣，因為 PCA 轉換是確定性的
        return price_pred
    
    # 不再需要 load_vae_encoder, freeze_encoder, unfreeze_encoder


def denormalize_surface(normalized_surface, mean, std):
    """Denormalize a surface using provided mean and std."""
    return normalized_surface * std + mean


def normalize_surface(surface, mean, std):
    """Normalize a surface using provided mean and std."""
    return (surface - mean) / std

# --------------------------
# Training Functions (PCA)
# --------------------------

def train_and_save_PCA(
    folder: str,
    latent_dim: int = 10,
    batch_size: int = 32,
    save_path: str = "pca_model.joblib"
):
    """
    "訓練" (擬合) 一個 PCA 模型並保存。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 只需要訓練數據來擬合 PCA
    train_loader, train_dataset = create_dataloader(
        folder, "post_vol_", "train", 
        batch_size=batch_size, 
        shuffle=False,  # 無需 shuffle 
        compute_stats=True # 確保 normalization stats 被加載/計算
    )

    # 確定輸入形狀
    first_batch, _, _ = next(iter(train_loader))
    input_shape = first_batch.shape[1:]  # (1, H, W)
    print(f"Detected input shape: {input_shape}")

    # 初始化並擬合 PCA
    pca_wrapper = PCAWrapper(n_components=latent_dim)
    pca_wrapper.fit(train_loader) # 傳遞 data_loader

    # 保存模型
    pca_wrapper.save(save_path)

    return pca_wrapper


def train_and_save_pricer_PCA(
    folder: str,
    product_type: str,
    pca_model_path: str, # 指向 .joblib
    latent_dim: int = 10,
    pricing_param_dim: int = 2,
    batch_size: int = 128,
    num_epochs: int = 150, # 只有一個訓練階段
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> tuple:
    """
    訓練一個使用預訓練 PCA 模型的 PricerPCA。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(folder, exist_ok=True)

    # 加載定價數據
    train_loader, train_dataset = create_pricing_dataloader(folder, product_type, "train", batch_size=batch_size, shuffle=True, compute_stats=True)
    test_loader, test_dataset = create_pricing_dataloader(folder, product_type, "test", batch_size=batch_size, shuffle=False, compute_stats=False)

    # 獲取形狀
    vol_first_batch, _, _ = next(iter(train_loader))
    vol_input_shape = vol_first_batch.shape[2:]  # (H, W)

    # 創建 PricerPCA 模型
    # 模型在初始化時會自動加載 pca_model_path
    model = PricerPCA(
        latent_dim=latent_dim, 
        pricing_param_dim=pricing_param_dim, 
        pca_model_path=pca_model_path,
        vol_input_shape=vol_input_shape
    ).to(device)

    # 優化器和調度器 (只優化 MLP 參數)
    optimizer = optim.Adam(model.pricing_mlp.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr * 0.01)

    train_losses, test_losses = [], []

    print("--- 開始訓練 Pricer (PCA) ---")
    
    # 只有一個訓練階段 (因為 PCA 部分是固定的)
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0

        for vol_surface, pricing_params, target_prices in train_loader:
            vol_surface = vol_surface.to(device)
            pricing_params = pricing_params.to(device)
            target_prices = target_prices.to(device)

            # Forward pass
            predicted_prices = model(vol_surface, pricing_params)

            # Compute loss
            loss = F.mse_loss(predicted_prices, target_prices)

            # Backward pass (只會更新 MLP 的權重)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * vol_surface.size(0)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        avg_train_loss = total_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        # 評估測試集
        model.eval()
        total_test_loss = 0.0
        with torch.no_grad():
            for vol_surface_test, pricing_params_test, target_prices_test in test_loader:
                vol_surface_test = vol_surface_test.to(device)
                pricing_params_test = pricing_params_test.to(device)
                target_prices_test = target_prices_test.to(device)

                predicted_prices_test = model(vol_surface_test, pricing_params_test)
                loss_test = F.mse_loss(predicted_prices_test, target_prices_test)
                total_test_loss += loss_test.item() * vol_surface_test.size(0)

        avg_test_loss = total_test_loss / len(test_loader.dataset)
        test_losses.append(avg_test_loss)

        print(f"Epoch {epoch+1}/{num_epochs} " f"| LR={current_lr:.2e} " f"| train_loss={avg_train_loss:.6f} " f"| test_loss={avg_test_loss:.6f}")

    # 保存模型狀態 (只保存 MLP 的權重) 和損失歷史
    state_path = os.path.join(folder, f"{product_type}_pricer_pca_state_dict.pt")
    torch.save(model.state_dict(), state_path)
    
    # 修正 loss 檔案名稱以區分 VAE
    np.save(os.path.join(folder, f"{product_type}_pricer_pca_train_losses.npy"), np.array(train_losses))
    np.save(os.path.join(folder, f"{product_type}_pricer_pca_test_losses.npy"), np.array(test_losses))

    print(f"Saved pricer (PCA) model state to {state_path}")
    print(f"Saved pricer (PCA) train/test losses to {folder}")

    return model, train_losses, test_losses


# --------------------------
# Plotting Functions (PCA)
# --------------------------

def plot_loss_curves(folder: str, product_type: str, model_type: str = "PCA"):
    """
    繪製 Pricer 模型的訓練/測試損失曲線。
    model_type: "PCA" or "VAE" (用於加載正確的檔案)
    """
    
    if model_type.upper() == "PCA":
        pricer_train_file = os.path.join(folder, f"{product_type}_pricer_pca_train_losses.npy")
        pricer_test_file = os.path.join(folder, f"{product_type}_pricer_pca_test_losses.npy")
        title_prefix = "Pricer (PCA)"
    elif model_type.upper() == "VAE":
         pricer_train_file = os.path.join(folder, f"{product_type}_pricer_train_losses.npy")
         pricer_test_file = os.path.join(folder, f"{product_type}_pricer_test_losses.npy")
         title_prefix = "Pricer (VAE)"
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    pricer_exists = os.path.exists(pricer_train_file) and os.path.exists(pricer_test_file)
    
    if not pricer_exists:
        print(f"Warning: Pricer loss files not found.")
        print(f"Looked for: {pricer_train_file}")
        print(f"Looked for: {pricer_test_file}")
        return

    pricer_train_losses = np.load(pricer_train_file)
    pricer_test_losses = np.load(pricer_test_file)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    pricer_epochs = np.arange(1, len(pricer_train_losses) + 1)
    ax.plot(pricer_epochs, pricer_train_losses, label="train", linewidth=2, color="red")
    ax.plot(pricer_epochs, pricer_test_losses, label="test", linewidth=2, linestyle="--", color="lightcoral")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title(f"{title_prefix} Loss Curves for {product_type}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{folder}/{product_type}_{model_type.lower()}_loss_curve.png", dpi=300)
    plt.show()
    plt.close()


def visualize_latent_distribution_PCA(
    model_path: str, 
    folder: str, 
    save_path: str = None
):
    """
    可視化 PCA 潛在變量的分佈。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加載訓練數據
    train_loader, _ = create_dataloader(folder, "post_vol_", "train", batch_size=32, shuffle=False, compute_stats=False)
    
    # 加載 PCA 模型
    pca_wrapper = PCAWrapper()
    pca_wrapper.load(model_path)
    latent_dim = pca_wrapper.n_components

    # 提取潛在表示
    latent_components = []
    with torch.no_grad():
        for x, _, _ in train_loader:
            x = x.to(device)
            z = pca_wrapper.transform(x) # (batch_size, latent_dim)
            latent_components.append(z.cpu().numpy())
            
    latent_components = np.concatenate(latent_components, axis=0)
    
    # --- 可視化 ---
    
    # 繪製前幾個主成分的直方圖
    num_dims_to_plot = min(10, latent_dim)
    num_rows = int(np.ceil(num_dims_to_plot / 2))
    fig, axes = plt.subplots(num_rows, 2, figsize=(12, num_rows * 3))
    axes = axes.flatten()

    for i in range(num_dims_to_plot):
        ax = axes[i]
        ax.hist(latent_components[:, i], bins=50, alpha=0.7, label=f"PC {i+1}")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Distribution of Principal Component {i+1}")
        ax.legend()

    # 隱藏多餘的子圖
    for i in range(num_dims_to_plot, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"PCA latent distribution saved to {save_path}")
    plt.show()

    # 打印統計信息
    print(f"\nLatent Component Statistics (using {latent_components.shape[0]} samples):")
    print(f"Mean: {np.mean(latent_components, axis=0)}")
    print(f"Std:  {np.std(latent_components, axis=0)}")
    print(f"Min:  {np.min(latent_components, axis=0)}")
    print(f"Max:  {np.max(latent_components, axis=0)}")

    return latent_components


def show_quote_date_reconstructions_PCA(
    folder: str,
    quote_dates: list,
    model_path: str, # .joblib file
    latent_dim: int,
    device: Optional[Union[str, torch.device]] = None,
    cmap: str = "rainbow",
    data_type: str = "train",
):
    """
    顯示特定日期的 PCA 重建效果。
    """
    if len(quote_dates) != 3:
        raise ValueError(f"Expected exactly 3 quote dates, got {len(quote_dates)}")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加載 normalization stats
    vol_stats_path = f"{folder}/vol_normalization_stats.npz"
    if os.path.exists(vol_stats_path):
        vol_stats = np.load(vol_stats_path)
        vol_mean = vol_stats["mean"]
        vol_std = vol_stats["std"]
        print(f"Loaded normalization stats: mean={vol_mean:.6f}, std={vol_std:.6f}")
    else:
        print("Warning: No normalization stats found. Reconstructions will be normalized")
        vol_mean = 0.0
        vol_std = 1.0

    # 加載 dataset
    _, dataset = create_dataloader(folder, "post_vol_", data_type, batch_size=1, shuffle=False, compute_stats=False)

    # 尋找日期
    quote_date_indices = []
    available_dates = dataset.quote_dates
    for target_date in quote_dates:
        if target_date in available_dates:
            idx = np.where(available_dates == target_date)[0][0]
            quote_date_indices.append(idx)
        else:
            print(f"Warning: {target_date} not found, skipping.")
            
    # 加載 PCA 模型
    pca_wrapper = PCAWrapper(n_components=latent_dim)
    pca_wrapper.load(model_path)

    # 產生重建
    figs = []
    actual_dates_used = []
    with torch.no_grad():
        for date_idx in quote_date_indices:
            surface, k_grid, T_grid = dataset[date_idx]
            actual_date = available_dates[date_idx]
            actual_dates_used.append(actual_date)
            x = surface.unsqueeze(0).to(device)  # (1, 1, H, W)

            # 轉換和逆轉換
            z = pca_wrapper.transform(x) # (1, latent_dim)
            recon = pca_wrapper.inverse_transform(z, output_shape=(1, 41, 20)) # (1, 1, H, W)

            x_np = x.squeeze(0).squeeze(0).cpu().numpy()
            recon_np = recon.squeeze(0).squeeze(0).cpu().numpy()

            # Denormalize
            x_denorm = x_np * vol_std + vol_mean
            recon_denorm = recon_np * vol_std + vol_mean
            figs.append((x_denorm, recon_denorm, z.squeeze(0).cpu().numpy(), actual_date, k_grid, T_grid))

    # --- 繪圖 ---
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    if len(figs) == 0:
        print("No valid dates found to plot.")
        return
        
    if len(figs) < 3:
        # 調整 axes 避免 index out of bound
        axes = axes[:len(figs)]

    title_suffix = " (denormalized)" if vol_mean != 0.0 else " (normalized)"

    for idx, fig_data in enumerate(figs):
        x_display, recon_display, z_np, actual_date, k_grid, T_grid = fig_data
        T_mesh, k_mesh = np.meshgrid(T_grid.numpy(), k_grid.numpy())
        
        # 確保 vmin/vmax 合理
        vmin = min(np.min(x_display), np.min(recon_display))
        vmax = max(np.max(x_display), np.max(recon_display))

        # Input
        ax_in = axes[idx, 0]
        pcm0 = ax_in.pcolormesh(k_mesh, T_mesh, x_display, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax_in.set_title(f"Input: {actual_date}{title_suffix}")
        ax_in.set_ylabel("k")
        ax_in.set_xlabel("T")
        fig.colorbar(pcm0, ax=ax_in)

        # Reconstruction
        ax_out = axes[idx, 1]
        pcm1 = ax_out.pcolormesh(k_mesh, T_mesh, recon_display, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        z_str = ", ".join([f"{v:+.2f}" for v in z_np[:3]]) + "..."
        ax_out.set_title(f"PCA Recon: {actual_date}\nz ≈ [{z_str}]")
        ax_out.set_ylabel("k")
        ax_out.set_xlabel("T")
        fig.colorbar(pcm1, ax=ax_out)

        # Difference
        ax_diff = axes[idx, 2]
        diff_display = x_display - recon_display
        diff_vmax = np.max(np.abs(diff_display))
        pcm2 = ax_diff.pcolormesh(k_mesh, T_mesh, diff_display, shading="auto", cmap="RdBu_r", vmin=-diff_vmax, vmax=diff_vmax)
        mse = np.mean(diff_display**2)
        ax_diff.set_title(f"Difference (MSE = {mse:.6f})")
        ax_diff.set_ylabel("k")
        ax_diff.set_xlabel("T")
        fig.colorbar(pcm2, ax=ax_diff)

    plt.tight_layout()
    plt.savefig(f"{folder}/pca_quote_date_reconstructions.png", dpi=300)
    plt.show()

def plot_predict_prices_from_vol_surface_and_params(
    folder: str,
    product_type: str,
    pricer_model_path: str, # .pt file for PricerPCA
    pca_model_path: str,    # .joblib file for PCAWrapper
    include_train: bool = True,
    latent_dim: int = 8,
    pricing_param_dim: int = 2,
    vol_input_shape: tuple = (41, 20),
    batch_size: int = 32,
    device: Optional[Union[str, torch.device]] = None,
) -> dict:
    """
    使用訓練好的 PricerPCA 模型預測價格並繪圖。
    (此函數與 VAE 版本幾乎相同，僅加載的模型不同)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加載訓練好的 PricerPCA 模型
    model = PricerPCA(
        latent_dim=latent_dim, 
        pricing_param_dim=pricing_param_dim, 
        pca_model_path=pca_model_path, # 傳入 .joblib 路徑
        vol_input_shape=vol_input_shape
    )
    model.load_state_dict(torch.load(pricer_model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Loaded trained pricer (PCA) model from {pricer_model_path}")

    # 加載用於反標準化的統計數據
    param_stats_path = f"{folder}/{product_type}_pricing_param_stats.npz"
    if os.path.exists(param_stats_path):
        stats = np.load(param_stats_path)
        price_params_mean = stats["params_mean"]
        price_params_std = stats["params_std"]
        price_mean = stats["price_mean"]
        price_std = stats["price_std"]
    else:
        raise FileNotFoundError(f"Pricing parameter stats not found at {param_stats_path}")

    def evaluate_dataset(data_type: str):
        """Helper function to evaluate a single dataset"""
        print(f"\nLoading {product_type} {data_type} data for price prediction...")
        loader, dataset = create_pricing_dataloader(folder, product_type, data_type, batch_size=batch_size, shuffle=False, compute_stats=False)

        predicted_prices_list = []
        target_prices_list = []
        pricing_params_norm_list = []

        print(f"Running predictions on {len(dataset)} {data_type} samples...")
        with torch.no_grad():
            for vol_surface, pricing_param, target_price in loader:
                vol_surface = vol_surface.to(device)
                pricing_param = pricing_param.to(device)
                target_price = target_price.to(device)

                # Forward pass through PricerPCA
                predicted_price = model(vol_surface, pricing_param)

                predicted_prices_list.append(predicted_price.cpu().numpy())
                target_prices_list.append(target_price.cpu().numpy())
                pricing_params_norm_list.append(pricing_param.cpu().numpy())

        predicted_prices = np.concatenate(predicted_prices_list, axis=0)
        target_prices = np.concatenate(target_prices_list, axis=0)
        pricing_params_norm = np.concatenate(pricing_params_norm_list, axis=0)

        # Denormalize
        pricing_params_denorm = pricing_params_norm * price_params_std + price_params_mean
        predicted_prices_denorm = predicted_prices * price_std + price_mean
        target_prices_denorm = target_prices * price_std + price_mean

        # Metrics
        mse = np.mean((predicted_prices_denorm - target_prices_denorm) ** 2)
        mae = np.mean(np.abs(predicted_prices_denorm - target_prices_denorm))
        rmse = np.sqrt(mse)
        ss_res = np.sum((target_prices_denorm - predicted_prices_denorm) ** 2)
        ss_tot = np.sum((target_prices_denorm - np.mean(target_prices_denorm)) ** 2)
        r2_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        print(f"\n{data_type.title()} Set Prediction Results (PCA Pricer):")
        print(f"R² Score: {r2_score:.6f}")
        print(f"MSE: {mse:.6f}, MAE: {mae:.6f}, RMSE: {rmse:.6f}")

        return {
            'predicted_prices': predicted_prices_denorm, 'target_prices': target_prices_denorm,
            'pricing_params': pricing_params_denorm, 'mse': mse, 'mae': mae, 'rmse': rmse, 'r2_score': r2_score
        }

    # --- 評估和繪圖 (與 VAE 版本相同) ---
    results = {}
    data_types = ['test']
    if include_train:
        data_types = ['train', 'test']
    for data_type in data_types:
        results[data_type] = evaluate_dataset(data_type)

    # 繪圖
    num_sets = len(data_types)
    fig, axes = plt.subplots(num_sets, 2, figsize=(15, 6 * num_sets), squeeze=False)

    for i, data_type in enumerate(data_types):
        result = results[data_type]
        predicted_flat = result['predicted_prices'].flatten()
        target_flat = result['target_prices'].flatten()
        
        # Scatter plot
        ax_scatter = axes[i, 0]
        ax_scatter.scatter(target_flat, predicted_flat, alpha=0.6, s=20, label=f"{data_type} data")
        min_price = min(np.min(target_flat), np.min(predicted_flat))
        max_price = max(np.max(target_flat), np.max(predicted_flat))
        ax_scatter.plot([min_price, max_price], [min_price, max_price], "r--", linewidth=2, label="Perfect Prediction")
        ax_scatter.set_xlabel("Ground Truth Price")
        ax_scatter.set_ylabel("Predicted Price")
        ax_scatter.set_title(f"{product_type} Price Prediction (PCA Pricer)\n{data_type.title()} Set (R² = {result['r2_score']:.4f})")
        ax_scatter.legend()
        ax_scatter.grid(True, alpha=0.3)
        stats_text = f"MSE: {result['mse']:.4f}\nMAE: {result['mae']:.4f}\nRMSE: {result['rmse']:.4f}"
        ax_scatter.text(0.05, 0.95, stats_text, transform=ax_scatter.transAxes, fontsize=10,
                        verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

        # Residual plot
        ax_residual = axes[i, 1]
        residuals = predicted_flat - target_flat
        ax_residual.scatter(target_flat, residuals, alpha=0.6, s=20, color="green")
        ax_residual.axhline(y=0, color="r", linestyle="--", linewidth=2)
        ax_residual.set_xlabel("Ground Truth Price")
        ax_residual.set_ylabel("Prediction Error (Predicted - Truth)")
        ax_residual.set_title(f"{data_type.title()} Set Residuals")
        ax_residual.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{folder}/{product_type}_pricer_pca_prediction_comparison.png", dpi=300)
    plt.show()
    plt.close()
    
    # ... (保存 .npz 檔案的邏輯可以照搬 VAE 版本) ...

    return results