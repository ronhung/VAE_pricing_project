# -------------------------------------------------------------------
# LDA_model.py
#
# 版本 2 (方法三):
# - 移除 K-Means。
# - LDA 的標籤 (y) 現在直接由 "真實價格" 分箱 (binning) 而來。
# - 這使得 LDA 模型必須針對每種產品 (AmericanPut, AsianCall) 單獨訓練。
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
import joblib 
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Optional, Union

# --------------------------
# Data Loading Functions
# (與 PCA_model.py 完全相同)
# (VolsurfaceDataset, create_dataloader, PricingDataset, create_pricing_dataloader)
# ... (為節省篇幅，此處省略，假設它們存在且與之前相同) ...
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
            self.prices_unnormalized = pricing_data["NPV"].copy() # 保存一份未標準化的
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
# Model Definitions (LDA v2)
# --------------------------

class LDAWrapper:
    """
    一個簡化的包裝器，只包裝 sklearn.discriminant_analysis.LinearDiscriminantAnalysis
    不再包含 K-Means。
    """
    def __init__(self, n_components=10):
        self.n_components = n_components
        self.lda = LinearDiscriminantAnalysis(n_components=self.n_components)
        self._is_fitted = False

    def fit(self, X_flat: np.ndarray, y_labels: np.ndarray):
        """
        直接使用 X 和 y (監督式) 擬合 LDA。
        X_flat: (n_samples, n_features)
        y_labels: (n_samples,)
        """
        print(f"Fitting LDA (n_components={self.n_components})...")
        print(f"X shape: {X_flat.shape}, y shape: {y_labels.shape}")
        
        # 檢查 n_components 是否有效
        n_classes = len(np.unique(y_labels))
        if self.n_components >= n_classes:
            new_ld = n_classes - 1
            print(f"Warning: n_components ({self.n_components}) >= n_classes ({n_classes}).")
            print(f"Reducing n_components to {new_ld}.")
            self.n_components = new_ld
            self.lda = LinearDiscriminantAnalysis(n_components=self.n_components)

        self.lda.fit(X_flat, y_labels)
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
            raise RuntimeError("LDA model is not fitted yet. Call .fit() first.")
        
        x_flat_np = x_batch_tensor.view(x_batch_tensor.size(0), -1).detach().cpu().numpy()
        z_np = self.lda.transform(x_flat_np)
        return torch.tensor(z_np, dtype=torch.float32, device=x_batch_tensor.device)

    def save(self, path: str):
        """使用 joblib 保存擬合好的 LDA 模型"""
        if not self._is_fitted:
            raise RuntimeError("Cannot save an unfitted model.")
        joblib.dump(self.lda, path)
        print(f"LDA model saved to {path}")

    def load(self, path: str):
            """使用 joblib 加載擬合好的模型"""
            if not os.path.exists(path):
                raise FileNotFoundError(f"LDA model file not found at {path}")
            
            self.lda = joblib.load(path)
            # BUGFIX: 使用 explained_variance_ratio_ 的長度來確定 component 數量
            # 因為 scalings_ 屬性可能不可靠
            self.n_components = len(self.lda.explained_variance_ratio_) # <--- 這是修正
            self._is_fitted = True
            # 更新日誌訊息以便區分
            print(f"LDA model loaded from {path} (n_components={self.n_components} from explained_variance_ratio_)")


class PricerLDA(nn.Module):
    """
    Pricer 模型，使用 *固定* 的 LDA 進行降維。
    (此結構基本不變)
    """
    def __init__(self, latent_dim: int, pricing_param_dim: int, lda_model_path: str, vol_input_shape: tuple = (41, 20)):
        super().__init__()
        self.latent_dim = latent_dim
        self.pricing_param_dim = pricing_param_dim
        self.vol_input_shape = vol_input_shape

        # 加載預先擬合的 LDA 模型
        self.lda_wrapper = LDAWrapper() 
        self.lda_wrapper.load(lda_model_path) 
        
        # 驗證加載的維度是否匹配
        if self.lda_wrapper.n_components != self.latent_dim:
            warnings.warn(
                f"Warning: requested latent_dim ({self.latent_dim}) "
                f"does not match loaded LDA n_components ({self.lda_wrapper.n_components}). "
                f"Using {self.lda_wrapper.n_components}."
            )
            self.latent_dim = self.lda_wrapper.n_components # 使用模型實際的維度

        # MLP for pricing: 
        # ** (方法一) 加入 Dropout 和簡化 MLP 來正則化 **
        self.pricing_mlp = nn.Sequential(
            nn.Linear(self.latent_dim + self.pricing_param_dim, 16),
            nn.ReLU(),
            nn.Dropout(p=0.3), # NEW: 加入 Dropout
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Dropout(p=0.3), # NEW: 加入 Dropout
            nn.Linear(16, 1),
        )

    def forward(self, vol_surface: torch.Tensor, pricing_params: torch.Tensor) -> torch.Tensor:
        # 1. 使用 LDA 轉換波動率曲面
        with torch.no_grad():
            z = self.lda_wrapper.transform(vol_surface) 
            z = z.to(vol_surface.device) 

        # 2. 串聯潛在向量和定價參數
        mlp_input = torch.cat([z, pricing_params], dim=1)

        # 3. 通過 MLP 進行定價
        price_pred = self.pricing_mlp(mlp_input)
        
        return price_pred
    

# --------------------------
# Training Functions (LDA v2)
# --------------------------

def train_and_save_LDA(
    folder: str,
    product_type: str, # NEW: 必須指定產品
    latent_dim: int = 10,
    n_bins: int = 11, # 要分箱的類別數
    save_path: str = "lda_model.joblib"
):
    """
    "訓練" (擬合) 一個 LDA 模型。
    標籤 y 來自於對 product_type 的*訓練集價格*進行分箱。
    """
    if n_bins <= latent_dim:
        raise ValueError(f"n_bins ({n_bins}) must be > latent_dim ({latent_dim})")
    
    # 1. 加載 *特定產品* 的 *訓練* 定價數據集
    print(f"Loading TRAINING data for {product_type} to create price-based labels...")
    # compute_param_stats=True 確保 stats 被加載/計算
    train_dataset = PricingDataset(folder, product_type, "train", compute_param_stats=True)
    
    # 2. 提取 X (標準化後的 vol surfaces)
    # 我們需要手動遍歷 Dataset 來提取所有 X
    surfaces_norm_list = [train_dataset[i][0] for i in range(len(train_dataset))]
    surfaces_norm_tensor = torch.stack(surfaces_norm_list)
    # (N, 1, H, W) -> (N, H*W)
    X_flat_norm = surfaces_norm_tensor.view(surfaces_norm_tensor.size(0), -1).cpu().numpy()
    
    # 3. 提取 y (未標準化的 prices) 並創建標籤
    prices_unnorm = train_dataset.prices_unnormalized # 從我們在 Dataset 中保存的副本獲取
    
    print(f"Binning {len(prices_unnorm)} prices into {n_bins} quantile-based bins...")
    
    try:
        # 使用 pd.qcut 進行分位數分箱
        # labels=False 產生整數標籤
        # duplicates='drop' 處理價格相同導致的 non-unique bin edges
        labels_y, bins = pd.qcut(prices_unnorm, q=n_bins, labels=False, retbins=True, duplicates='drop')
        
        actual_bins = len(np.unique(labels_y)) # 實際產生的 bin 數量
        print(f"Actual unique bins created (due to duplicate edges): {actual_bins}")

        # 檢查實際 bin 數是否足夠
        if actual_bins <= latent_dim:
            print(f"Error: Binning resulted in {actual_bins} unique bins, which is <= latent_dim ({latent_dim}).")
            print("LDA n_components must be < n_classes.")
            # 嘗試減少 latent_dim
            new_ld = actual_bins - 1
            if new_ld < 1:
                raise ValueError(f"Cannot proceed: Only {actual_bins} unique price bins found. Cannot fit LDA.")
            print(f"Reducing latent_dim to {new_ld}.")
            latent_dim = new_ld # 更新 latent_dim
            
    except ValueError as e:
        print(f"Error during qcut binning: {e}")
        print("This can happen if many prices are identical (e.g., 0).")
        print(f"Price stats: min={np.min(prices_unnorm)}, max={np.max(prices_unnorm)}, median={np.median(prices_unnorm)}")
        raise

    print(f"Bin distribution: {np.bincount(labels_y)}")

    # 4. 初始化並擬合 LDA
    lda_wrapper = LDAWrapper(n_components=latent_dim)
    lda_wrapper.fit(X_flat_norm, labels_y)
    
    # 5. 保存模型
    lda_wrapper.save(save_path)
    print(f"Saved product-specific LDA model to {save_path}")
    
    return lda_wrapper


def train_and_save_pricer_LDA(
    folder: str,
    product_type: str,
    lda_model_path: str, # 指向 product-specific .joblib
    latent_dim: int = 10,
    pricing_param_dim: int = 2,
    batch_size: int = 128,
    num_epochs: int = 150, 
    lr: float = 1e-3,
    weight_decay: float = 1e-3, # (方法一) 提高 L2 正則化
) -> tuple:
    """
    訓練一個使用預訓練 LDA 模型的 PricerLDA。
    (此函數與 v1 版本幾乎相同，但我們提高了 weight_decay)
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
    
    # ** (方法一) 應用 Early Stopping 的準備 **
    best_test_loss = float('inf')
    patience = 25 # 連續 25 epochs 測試集沒改善就停止
    patience_counter = 0
    best_model_state = None

    # 優化器和調度器
    optimizer = optim.Adam(model.pricing_mlp.parameters(), lr=lr, weight_decay=weight_decay) # 使用了更新的 weight_decay
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr * 0.01)

    train_losses, test_losses = [], []

    print(f"--- 開始訓練 Pricer (LDA v2) for {product_type} ---")
    print(f"Using Dropout (p=0.3) and Weight Decay (L2={weight_decay})")
    
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

        # 評估測試集 (for early stopping)
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

        # ** (方法一) Early Stopping 邏輯 **
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            patience_counter = 0
            # 保存最佳模型
            best_model_state = model.state_dict()
            print(f"  -> New best test loss: {best_test_loss:.6f}. Saving model state.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  -> Test loss did not improve for {patience} epochs. Stopping early.")
                break

    # 保存 *最佳* 模型狀態
    if best_model_state is None: # 如果
        best_model_state = model.state_dict() # 至少保存最後一個

    state_path = os.path.join(folder, f"{product_type}_pricer_lda_state_dict.pt")
    torch.save(best_model_state, state_path)
    
    np.save(os.path.join(folder, f"{product_type}_pricer_lda_train_losses.npy"), np.array(train_losses))
    np.save(os.path.join(folder, f"{product_type}_pricer_lda_test_losses.npy"), np.array(test_losses))

    print(f"Saved BEST pricer (LDA) model state to {state_path} (best test loss: {best_test_loss:.6f})")

    return model, train_losses, test_losses


# --------------------------
# Plotting Functions (LDA v2)
# --------------------------

# plot_loss_curves 保持不變 (已在 v1 中更新)
def plot_loss_curves(folder: str, product_type: str, model_type: str = "LDA"):
    # ... (與 v1 相同) ...
    if model_type.upper() == "PCA":
        pricer_train_file = os.path.join(folder, f"{product_type}_pricer_pca_train_losses.npy")
        pricer_test_file = os.path.join(folder, f"{product_type}_pricer_pca_test_losses.npy")
        title_prefix = "Pricer (PCA)"
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

    pricer_exists = os.path.exists(pricer_train_file) and os.path.exists(pricer_test_file)
    
    if not pricer_exists:
        print(f"Warning: Pricer loss files not found.")
        print(f"Looked for: {pricer_train_file}")
        print(f"Looked for: {pricer_test_file}")
        return

    pricer_train_losses = np.load(pricer_train_file)
    pricer_test_losses = np.load(pricer_test_file)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    # 繪製到實際停止的 epoch
    epochs_ran = np.arange(1, len(pricer_train_losses) + 1)
    ax.plot(epochs_ran, pricer_train_losses, label="train", linewidth=2, color="blue")
    ax.plot(epochs_ran, pricer_test_losses, label="test", linewidth=2, linestyle="--", color="lightblue")
    
    # 標記 Early Stop 點
    if len(pricer_train_losses) < 150: # 假設 num_epochs=150
        best_epoch = np.argmin(pricer_test_losses)
        ax.axvline(x=best_epoch + 1, color='red', linestyle='--', linewidth=1, label=f'Early Stop @ Ep {best_epoch+1}')

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
    product_type: str, # NEW: 需要知道為哪個產品
    save_path: str = None
):
    """
    可視化 LDA 潛在變量的分佈。
    數據源現在是特定產品的 PricingDataset。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加載 *特定產品* 的 *訓練* 定價數據集
    print(f"Loading TRAINING data for {product_type} to visualize distribution...")
    train_dataset = PricingDataset(folder, product_type, "train", compute_param_stats=False)
    
    # 2. 提取 X (標準化後的 vol surfaces)
    surfaces_norm_list = [train_dataset[i][0] for i in range(len(train_dataset))]
    surfaces_norm_tensor = torch.stack(surfaces_norm_list) # (N, 1, H, W)
    
    # 3. 加載 LDA 模型
    lda_wrapper = LDAWrapper()
    lda_wrapper.load(model_path)
    latent_dim = lda_wrapper.n_components

    # 4. 提取潛在表示
    latent_components = []
    # 由於數據量可能很大，我們分批處理
    batch_size = 64
    num_samples = surfaces_norm_tensor.size(0)
    
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            x_batch = surfaces_norm_tensor[i : i+batch_size].to(device)
            z_batch = lda_wrapper.transform(x_batch) # (batch, latent_dim)
            latent_components.append(z_batch.cpu().numpy())
            
    latent_components = np.concatenate(latent_components, axis=0)
    
    # --- 可視化 (與 v1 相同) ---
    num_dims_to_plot = min(10, latent_dim)
    num_rows = int(np.ceil(num_dims_to_plot / 2))
    fig, axes = plt.subplots(num_rows, 2, figsize=(12, num_rows * 3))
    axes = axes.flatten()

    for i in range(num_dims_to_plot):
        ax = axes[i]
        ax.hist(latent_components[:, i], bins=50, alpha=0.7, label=f"LD {i+1}", color="green")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Distribution of Latent Discriminant {i+1} (for {product_type})")
        ax.legend()

    for i in range(num_dims_to_plot, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"LDA latent distribution saved to {save_path}")
    plt.show()

    return latent_components


# plot_predict_prices... 函數重命名以匹配 (plot_predict_prices..._LDA)
# 但其內部邏輯與 v1 相同，此處為清楚起見重命名
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
    (此函數邏輯與 v1 相同)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加載訓練好的 PricerLDA 模型
    model = PricerLDA(
        latent_dim=latent_dim, 
        pricing_param_dim=pricing_param_dim, 
        lda_model_path=lda_model_path, 
        vol_input_shape=vol_input_shape
    )
    # 我們加載 state_dict，PricerLDA 的 __init__ 已經加載了 lda_wrapper
    model.load_state_dict(torch.load(pricer_model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Loaded trained pricer (LDA) model from {pricer_model_path}")

    # ... (其餘的 evaluate_dataset 和 繪圖邏輯與 v1 完全相同) ...
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
        """Helper function (identical to v1)"""
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

        print(f"\n{data_type.title()} Set Prediction Results (LDA Pricer v2):")
        print(f"R² Score: {r2_score:.6f}")
        print(f"MSE: {mse:.6f}, MAE: {mae:.6f}, RMSE: {rmse:.6f}")

        return {
            'predicted_prices': predicted_prices_denorm, 'target_prices': target_prices_denorm,
            'pricing_params': pricing_params_denorm, 'mse': mse, 'mae': mae, 'rmse': rmse, 'r2_score': r2_score
        }

    # --- 評估和繪圖 ---
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
        
        ax_scatter = axes[i, 0]
        ax_scatter.scatter(target_flat, predicted_flat, alpha=0.6, s=20, label=f"{data_type} data")
        min_price = min(np.min(target_flat), np.min(predicted_flat))
        max_price = max(np.max(target_flat), np.max(predicted_flat))
        ax_scatter.plot([min_price, max_price], [min_price, max_price], "r--", linewidth=2, label="Perfect Prediction")
        ax_scatter.set_xlabel("Ground Truth Price")
        ax_scatter.set_ylabel("Predicted Price")
        ax_scatter.set_title(f"{product_type} Price Prediction (LDA Pricer v2)\n{data_type.title()} Set (R² = {result['r2_score']:.4f})")
        ax_scatter.legend()
        ax_scatter.grid(True, alpha=0.3)
        stats_text = f"MSE: {result['mse']:.4f}\nMAE: {result['mae']:.4f}\nRMSE: {result['rmse']:.4f}"
        ax_scatter.text(0.05, 0.95, stats_text, transform=ax_scatter.transAxes, fontsize=10,
                        verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

        ax_residual = axes[i, 1]
        residuals = predicted_flat - target_flat
        ax_residual.scatter(target_flat, residuals, alpha=0.6, s=20, color="green")
        ax_residual.axhline(y=0, color="r", linestyle="--", linewidth=2)
        ax_residual.set_xlabel("Ground Truth Price")
        ax_residual.set_ylabel("Prediction Error (Predicted - Truth)")
        ax_residual.set_title(f"{data_type.title()} Set Residuals")
        ax_residual.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{folder}/{product_type}_pricer_lda_prediction_comparison.png", dpi=300)
    plt.show()
    plt.close()
    
    return results