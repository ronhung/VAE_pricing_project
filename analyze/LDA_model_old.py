# -------------------------------------------------------------------
# LDA_model.py
#
# 仿造 VAE_model.py / PCA_model.py 的結構，
# 但使用 K-Means (K=11) 產生標籤，
# 並使用 sklearn.discriminant_analysis.LinearDiscriminantAnalysis (LDA)
# 替換 PCA 的轉換。
# -------------------------------------------------------------------

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ML_analyze import * # 假設 ML_analyze.py 存在
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis # NEW
from sklearn.cluster import KMeans # NEW
import joblib 
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Optional, Union

# --------------------------
# Data Loading Functions
# (與 PCA_model.py 完全相同)
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
# Model Definitions (LDA)
# --------------------------

class LDAWrapper:
    """
    一個包裝 K-Means 和 LDA 的類。
    1. 使用 K-Means (非監督式) 從 X 創建標籤 y。
    2. 使用 X 和 y (監督式) 擬合 LDA。
    """
    def __init__(self, n_components=10, n_clusters=11):
        if n_clusters <= n_components:
            raise ValueError(f"n_clusters ({n_clusters}) must be > n_components ({n_components}). "
                             f"Rule: n_components <= n_clusters - 1")
        
        self.n_components = n_components
        self.n_clusters = n_clusters
        
        # 1. K-Means for labeling
        self.kmeans = KMeans(
            n_clusters=self.n_clusters, 
            init='k-means++', 
            n_init=10, 
            random_state=42
        )
        
        # 2. LDA for dimensionality reduction
        self.lda = LinearDiscriminantAnalysis(n_components=self.n_components)
        
        self._is_fitted = False

    def _load_data_for_fitting(self, data_loader: DataLoader) -> np.ndarray:
        """從 dataloader 加載所有數據並展平 (X)"""
        all_surfaces = []
        print("Gathering all surfaces for K-Means/LDA fitting...")
        for batch, _, _ in data_loader:
            # batch shape is (batch_size, 1, H, W)
            all_surfaces.append(batch.numpy())
        
        surfaces_np = np.concatenate(all_surfaces, axis=0)
        # 展平為 (num_samples, H*W)
        surfaces_flat = surfaces_np.reshape(surfaces_np.shape[0], -1)
        print(f"Data shape for fitting: {surfaces_flat.shape}")
        return surfaces_flat

    def fit(self, data_loader: DataLoader):
        """從 dataloader 擬合 K-Means 和 LDA 模型"""
        
        # 1. 加載所有數據 (X)
        surfaces_flat = self._load_data_for_fitting(data_loader)
        
        # 2. 擬合 K-Means 並產生標籤 (y)
        print(f"Fitting K-Means (K={self.n_clusters}) to generate labels...")
        self.kmeans.fit(surfaces_flat)
        labels_y = self.kmeans.labels_
        print("K-Means fitting complete.")
        print(f"Cluster distribution: {np.bincount(labels_y)}")

        # 3. 擬合 LDA (X, y)
        print(f"Fitting LDA (n_components={self.n_components}) using generated labels...")
        self.lda.fit(surfaces_flat, labels_y)
        self._is_fitted = True
        print("LDA fitting complete.")
        
        # 打印解釋的方差
        explained_variance = self.lda.explained_variance_ratio_ * 100
        total_explained_variance = np.sum(explained_variance)
        print(f"Explained variance per component (%): {explained_variance}")
        print(f"Total explained variance by {self.n_components} components: {total_explained_variance:.2f}%")

    def transform(self, x_batch_tensor: torch.Tensor) -> torch.Tensor:
        """
        將一批波動率曲面 (tensor) 轉換為潛在向量 (tensor)。
        x_batch_tensor: (batch_size, 1, H, W)
        """
        if not self._is_fitted:
            raise RuntimeError("LDA/K-Means models are not fitted yet. Call .fit() first.")
        
        # 展平並轉為 numpy
        x_flat_np = x_batch_tensor.view(x_batch_tensor.size(0), -1).detach().cpu().numpy()
        
        # 轉換 (LDA transform 只需要 X)
        z_np = self.lda.transform(x_flat_np)
        
        # 轉回 tensor
        return torch.tensor(z_np, dtype=torch.float32, device=x_batch_tensor.device)

    # *** 注意：LDA 沒有 inverse_transform ***
    # `show_quote_date_reconstructions` 函數將無法實現

    def save(self, path: str):
        """使用 joblib 保存擬合好的 K-Means 和 LDA 模型"""
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted models.")
        
        models_to_save = {
            "kmeans": self.kmeans,
            "lda": self.lda
        }
        joblib.dump(models_to_save, path)
        print(f"LDA/K-Means models saved to {path}")

    def load(self, path: str):
        """使用 joblib 加載擬合好的模型"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"LDA/K-Means model file not found at {path}")
        
        models = joblib.load(path)
        self.kmeans = models["kmeans"]
        self.lda = models["lda"]
        
        # 從加載的模型中更新參數
        self.n_clusters = self.kmeans.n_clusters
        # lda.scalings_ 存儲投影向量，其 col 數即為 components 數
        self.n_components = self.lda.scalings_.shape[1] 
        self._is_fitted = True
        
        print(f"LDA/K-Means models loaded from {path}")
        print(f"  -> n_components = {self.n_components} (from loaded LDA)")
        print(f"  -> n_clusters = {self.n_clusters} (from loaded K-Means)")


class PricerLDA(nn.Module):
    """
    Pricer 模型，使用 *固定* 的 LDA 進行降維。
    (結構與 PricerPCA 相同)
    """
    def __init__(self, latent_dim: int, pricing_param_dim: int, lda_model_path: str, vol_input_shape: tuple = (41, 20)):
        super().__init__()
        self.latent_dim = latent_dim
        self.pricing_param_dim = pricing_param_dim
        self.vol_input_shape = vol_input_shape

        # 加載預先擬合的 LDA/K-Means 模型
        self.lda_wrapper = LDAWrapper() # 初始化
        self.lda_wrapper.load(lda_model_path) # 加載
        
        # 驗證加載的維度是否匹配
        if self.lda_wrapper.n_components != self.latent_dim:
            warnings.warn(
                f"Warning: requested latent_dim ({self.latent_dim}) "
                f"does not match loaded LDA n_components ({self.lda_wrapper.n_components}). "
                f"Using {self.lda_wrapper.n_components}."
            )
            self.latent_dim = self.lda_wrapper.n_components

        # MLP for pricing: 
        self.pricing_mlp = nn.Sequential(
            nn.Linear(self.latent_dim + self.pricing_param_dim, 16),
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
            vol_surface: (batch_size, 1, H, W)
            pricing_params: (batch_size, pricing_param_dim)
        """
        # 1. 使用 LDA 轉換波動率曲面
        with torch.no_grad():
            z = self.lda_wrapper.transform(vol_surface) 
            z = z.to(vol_surface.device) 

        # 2. 串聯潛在向量和定價參數
        mlp_input = torch.cat([z, pricing_params], dim=1)

        # 3. 通過 MLP 進行定價
        price_pred = self.pricing_mlp(mlp_input)
        
        return price_pred
    
# (denormalize/normalize 函數保持不變)
def denormalize_surface(normalized_surface, mean, std): ...
def normalize_surface(surface, mean, std): ...

# --------------------------
# Training Functions (LDA)
# --------------------------

def train_and_save_LDA(
    folder: str,
    latent_dim: int = 10,
    n_clusters: int = 11, # K-Means clusters
    batch_size: int = 32,
    save_path: str = "lda_kmeans_model.joblib"
):
    """
    "訓練" (擬合) 一個 K-Means + LDA 模型並保存。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 只需要訓練數據來擬合
    train_loader, train_dataset = create_dataloader(
        folder, "post_vol_", "train", 
        batch_size=batch_size, 
        shuffle=False,  # 必須是 False 才能重現 K-Means
        compute_stats=True
    )

    # 確定輸入形狀
    first_batch, _, _ = next(iter(train_loader))
    input_shape = first_batch.shape[1:]  # (1, H, W)
    print(f"Detected input shape: {input_shape}")

    # 初始化並擬合 LDAWrapper
    lda_wrapper = LDAWrapper(n_components=latent_dim, n_clusters=n_clusters)
    lda_wrapper.fit(train_loader) # 傳遞 data_loader

    # 保存模型
    lda_wrapper.save(save_path)

    return lda_wrapper


def train_and_save_pricer_LDA(
    folder: str,
    product_type: str,
    lda_model_path: str, # 指向 .joblib
    latent_dim: int = 10,
    pricing_param_dim: int = 2,
    batch_size: int = 128,
    num_epochs: int = 150, 
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> tuple:
    """
    訓練一個使用預訓練 LDA 模型的 PricerLDA。
    (此函數與 PCA 版本幾乎相同)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(folder, exist_ok=True)

    # 加載定價數據
    train_loader, train_dataset = create_pricing_dataloader(folder, product_type, "train", batch_size=batch_size, shuffle=True, compute_stats=True)
    test_loader, test_dataset = create_pricing_dataloader(folder, product_type, "test", batch_size=batch_size, shuffle=False, compute_stats=False)

    vol_first_batch, _, _ = next(iter(train_loader))
    vol_input_shape = vol_first_batch.shape[2:]  # (H, W)

    # 創建 PricerLDA 模型
    model = PricerLDA(
        latent_dim=latent_dim, 
        pricing_param_dim=pricing_param_dim, 
        lda_model_path=lda_model_path,
        vol_input_shape=vol_input_shape
    ).to(device)

    # 優化器和調度器 (只優化 MLP 參數)
    optimizer = optim.Adam(model.pricing_mlp.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr * 0.01)

    train_losses, test_losses = [], []

    print("--- 開始訓練 Pricer (LDA) ---")
    
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0

        for vol_surface, pricing_params, target_prices in train_loader:
            vol_surface = vol_surface.to(device)
            pricing_params = pricing_params.to(device)
            target_prices = target_prices.to(device)

            predicted_prices = model(vol_surface, pricing_params)
            loss = F.mse_loss(predicted_prices, target_prices)

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

    # CHANGED: 儲存檔案名稱
    state_path = os.path.join(folder, f"{product_type}_pricer_lda_state_dict.pt")
    torch.save(model.state_dict(), state_path)
    
    np.save(os.path.join(folder, f"{product_type}_pricer_lda_train_losses.npy"), np.array(train_losses))
    np.save(os.path.join(folder, f"{product_type}_pricer_lda_test_losses.npy"), np.array(test_losses))

    print(f"Saved pricer (LDA) model state to {state_path}")
    print(f"Saved pricer (LDA) train/test losses to {folder}")

    return model, train_losses, test_losses


# --------------------------
# Plotting Functions (LDA)
# --------------------------

def plot_loss_curves(folder: str, product_type: str, model_type: str = "PCA"):
    """
    繪製 Pricer 模型的訓練/測試損失曲線。
    model_type: "PCA", "VAE", or "LDA"
    """
    
    if model_type.upper() == "PCA":
        pricer_train_file = os.path.join(folder, f"{product_type}_pricer_pca_train_losses.npy")
        pricer_test_file = os.path.join(folder, f"{product_type}_pricer_pca_test_losses.npy")
        title_prefix = "Pricer (PCA)"
    # NEW: Added LDA case
    elif model_type.upper() == "LDA":
        pricer_train_file = os.path.join(folder, f"{product_type}_pricer_lda_train_losses.npy")
        pricer_test_file = os.path.join(folder, f"{product_type}_pricer_lda_test_losses.npy")
        title_prefix = "Pricer (LDA)"
    elif model_type.upper() == "VAE":
         pricer_train_file = os.path.join(folder, f"{product_type}_pricer_train_losses.npy")
         pricer_test_file = os.path.join(folder, f"{product_type}_pricer_test_losses.npy")
         title_prefix = "Pricer (VAE)"
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # ... (其餘繪圖代碼與 PCA_model.py 完全相同) ...
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
    ax.plot(pricer_epochs, pricer_train_losses, label="train", linewidth=2, color="blue") # Changed color
    ax.plot(pricer_epochs, pricer_test_losses, label="test", linewidth=2, linestyle="--", color="lightblue") # Changed color
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


def visualize_latent_distribution_LDA(
    model_path: str, # .joblib file
    folder: str, 
    save_path: str = None
):
    """
    可視化 LDA 潛在變量 (discriminant components) 的分佈。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加載訓練數據
    train_loader, _ = create_dataloader(folder, "post_vol_", "train", batch_size=32, shuffle=False, compute_stats=False)
    
    # 加載 LDA/K-Means 模型
    lda_wrapper = LDAWrapper()
    lda_wrapper.load(model_path)
    latent_dim = lda_wrapper.n_components

    # 提取潛在表示
    latent_components = []
    with torch.no_grad():
        for x, _, _ in train_loader:
            x = x.to(device)
            z = lda_wrapper.transform(x) # (batch_size, latent_dim)
            latent_components.append(z.cpu().numpy())
            
    latent_components = np.concatenate(latent_components, axis=0)
    
    # --- 可視化 (與 PCA 版本相同) ---
    num_dims_to_plot = min(10, latent_dim)
    num_rows = int(np.ceil(num_dims_to_plot / 2))
    fig, axes = plt.subplots(num_rows, 2, figsize=(12, num_rows * 3))
    axes = axes.flatten()

    for i in range(num_dims_to_plot):
        ax = axes[i]
        ax.hist(latent_components[:, i], bins=50, alpha=0.7, label=f"LD {i+1}", color="green")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Distribution of Latent Discriminant {i+1}")
        ax.legend()

    for i in range(num_dims_to_plot, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"LDA latent distribution saved to {save_path}")
    plt.show()

    # 打印統計信息
    print(f"\nLatent Component Statistics (using {latent_components.shape[0]} samples):")
    print(f"Mean: {np.mean(latent_components, axis=0)}")
    print(f"Std:  {np.std(latent_components, axis=0)}")
    print(f"Min:  {np.min(latent_components, axis=0)}")
    print(f"Max:  {np.max(latent_components, axis=0)}")

    return latent_components


# *** OMITTED: show_quote_date_reconstructions_LDA ***
# (LDA does not support inverse_transform)


def plot_predict_prices_from_vol_surface_and_params_LDA(
    folder: str,
    product_type: str,
    pricer_model_path: str, # .pt file for PricerLDA
    lda_model_path: str,    # .joblib file for LDAWrapper
    include_train: bool = True,
    latent_dim: int = 10,
    pricing_param_dim: int = 2,
    vol_input_shape: tuple = (41, 20),
    batch_size: int = 32,
    device: Optional[Union[str, torch.device]] = None,
) -> dict:
    """
    使用訓練好的 PricerLDA 模型預測價格並繪圖。
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加載訓練好的 PricerLDA 模型
    model = PricerLDA(
        latent_dim=latent_dim, 
        pricing_param_dim=pricing_param_dim, 
        lda_model_path=lda_model_path, # 傳入 .joblib 路徑
        vol_input_shape=vol_input_shape
    )
    model.load_state_dict(torch.load(pricer_model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Loaded trained pricer (LDA) model from {pricer_model_path}")

    # 加載用於反標準化的統計數據 (與 PCA 版本相同)
    param_stats_path = f"{folder}/{product_type}_pricing_param_stats.npz"
    # ... (stats loading logic) ...
    if os.path.exists(param_stats_path):
        stats = np.load(param_stats_path)
        price_params_mean = stats["params_mean"]
        price_params_std = stats["params_std"]
        price_mean = stats["price_mean"]
        price_std = stats["price_std"]
    else:
        raise FileNotFoundError(f"Pricing parameter stats not found at {param_stats_path}")

    def evaluate_dataset(data_type: str):
        """Helper function (identical to PCA version)"""
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

                # Forward pass through PricerLDA
                predicted_price = model(vol_surface, pricing_param)

                predicted_prices_list.append(predicted_price.cpu().numpy())
                target_prices_list.append(target_price.cpu().numpy())
                pricing_params_norm_list.append(pricing_param.cpu().numpy())
        
        # ... (Denormalize and Metrics calculation, identical to PCA version) ...
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

        print(f"\n{data_type.title()} Set Prediction Results (LDA Pricer):")
        print(f"R² Score: {r2_score:.6f}")
        print(f"MSE: {mse:.6f}, MAE: {mae:.6f}, RMSE: {rmse:.6f}")

        return {
            'predicted_prices': predicted_prices_denorm, 'target_prices': target_prices_denorm,
            'pricing_params': pricing_params_denorm, 'mse': mse, 'mae': mae, 'rmse': rmse, 'r2_score': r2_score
        }

    # --- 評估和繪圖 (與 PCA 版本相同) ---
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
        # CHANGED: Title
        ax_scatter.set_title(f"{product_type} Price Prediction (LDA Pricer)\n{data_type.title()} Set (R² = {result['r2_score']:.4f})")
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
    # CHANGED: save file name
    plt.savefig(f"{folder}/{product_type}_pricer_lda_prediction_comparison.png", dpi=300)
    plt.show()
    plt.close()
    
    return results