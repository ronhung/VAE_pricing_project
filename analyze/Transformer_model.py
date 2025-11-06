# -------------------------------------------------------------------
# Transformer_model.py
#
# 使用 Transformer 架構作為 Autoencoder (TAE) 進行降維。
# 1. 將 41x20 的曲面視為一個長度 41 (strike) 的序列，
#    每個元素有 20 個特徵 (tenor)。
# 2. TransformerEncoder 學習壓縮此序列。
# 3. PricerTransformer 使用此壓縮後的潛在向量 z。
# -------------------------------------------------------------------

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Optional, Union
import math # Transformer 需要

# --------------------------
# Data Loading Functions
# (與 PCA_model.py 完全相同)
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
            self.norm_mean = np.mean(self.surfaces_grid)
            self.norm_std = np.std(self.surfaces_grid)
            np.savez(self.stats_path, mean=self.norm_mean, std=self.norm_std)
            print(f"Computed normalization stats: mean={self.norm_mean:.6f}, std={self.norm_std:.6f}")
            print(f"Saved normalization stats to {self.stats_path}")
        else:
            if os.path.exists(self.stats_path):
                stats = np.load(self.stats_path)
                self.norm_mean = stats["mean"].item()
                self.norm_std = stats["std"].item()
                print(f"Loaded normalization stats: mean={self.norm_mean:.6f}, std={self.norm_std:.6f}")
            else:
                print(f"Warning: No normalization stats found at {self.stats_path}, computing from current data")
                self.norm_mean = np.mean(self.surfaces_grid)
                self.norm_std = np.std(self.surfaces_grid)
                np.savez(self.stats_path, mean=self.norm_mean, std=self.norm_std)

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
        pricing_data_path = f"{folder}/{product_type}_pricing_data_{data_type}.npz"
        print(f"Loading pricing data from {pricing_data_path}")
        pricing_data = np.load(pricing_data_path, allow_pickle=True)

        if product_type == "AmericanPut" or product_type == "AsianCall" or product_type == "AsianPut":
            self.quote_dates = pricing_data["quote_dates"]
            self.vol_surfaces = pricing_data["vol_surfaces"]
            self.price_params = np.column_stack((pricing_data["K"], pricing_data["T"]))
            self.prices = pricing_data["NPV"]
            self.spot_prices = pricing_data["UNDERLYING_LAST"] # <--- 1. 讀取 S0
        else:
            raise ValueError(f"Unsupported product type: {product_type}")

        vol_stats_path = f"{folder}/vol_normalization_stats.npz"
        if os.path.exists(vol_stats_path):
            stats = np.load(vol_stats_path)
            self.norm_mean = stats["mean"].item()
            self.norm_std = stats["std"].item()
        else:
            raise FileNotFoundError(f"Normalization stats not found at {vol_stats_path}")

        self.vol_surfaces = (self.vol_surfaces - self.norm_mean) / self.norm_std

        param_stats_path = f"{folder}/{product_type}_pricing_param_stats.npz"
        if compute_param_stats:
            self.price_params_mean = np.mean(self.price_params, axis=0)
            self.price_params_std = np.std(self.price_params, axis=0)
            self.price_mean = np.mean(self.prices)
            self.price_std = np.std(self.prices)
            self.spot_mean = np.mean(self.spot_prices) # <--- 2. 計算 S0 統計
            self.spot_std = np.std(self.spot_prices)   # <--- 2. 計算 S0 統計

            price_param_stats = {
                "params_mean": self.price_params_mean, "params_std": self.price_params_std,
                "price_mean": self.price_mean, "price_std": self.price_std,
                "spot_mean": self.spot_mean, "spot_std": self.spot_std, # <--- 3. 儲存 S0 統計
            }
            np.savez(param_stats_path, **price_param_stats)
        else:
            if os.path.exists(param_stats_path):
                stats = np.load(param_stats_path)
                self.price_params_mean = stats["params_mean"]
                self.price_params_std = stats["params_std"]
                self.price_mean = stats["price_mean"]
                self.price_std = stats["price_std"]
                self.spot_mean = stats.get("spot_mean", np.mean(self.spot_prices)) # <--- 4. 載入 S0 統計 (兼容舊檔)
                self.spot_std = stats.get("spot_std", np.std(self.spot_prices))   # <--- 4. 載入 S0 統計 (兼容舊檔)
            else:
                warnings.warn(f"No pricing parameter stats found at {param_stats_path}, computing from current data")
                self.price_params_mean = np.mean(self.price_params, axis=0)
                self.price_params_std = np.std(self.price_params, axis=0)
                self.price_mean = np.mean(self.prices)
                self.price_std = np.std(self.prices)
                self.spot_mean = np.mean(self.spot_prices) # <--- 4. (Fallback)
                self.spot_std = np.std(self.spot_prices)   # <--- 4. (Fallback)

        self.price_params = (self.price_params - self.price_params_mean) / self.price_params_std
        self.prices = (self.prices - self.price_mean) / self.price_std
        self.spot_prices = (self.spot_prices - self.spot_mean) / self.spot_std # <--- 5. 標準化 S0

    def __len__(self):
        return len(self.quote_dates)

    def __getitem__(self, idx):
        vol_surface = torch.from_numpy(self.vol_surfaces[idx]).float().unsqueeze(0)  # add channel dimension
        pricing_param = torch.tensor(self.price_params[idx], dtype=torch.float32)
        spot_price = torch.tensor(self.spot_prices[idx], dtype=torch.float32).unsqueeze(0) # <--- 6. 取得 S0
        price = torch.tensor(self.prices[idx], dtype=torch.float32).unsqueeze(0)  # make it (1,) for consistency
        return vol_surface, pricing_param, spot_price, price # <--- 7. 回傳 4 個項目


def create_pricing_dataloader(folder, label, data_type, batch_size=32, shuffle=True, compute_stats=False):
    """Create a DataLoader for pricing data."""
    dataset = PricingDataset(folder, label, data_type, compute_param_stats=compute_stats)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle), dataset

# --------------------------
# Model Definitions (Transformer)
# --------------------------

class PositionalEncoding(nn.Module):
    """
    標準的 Transformer 位置編碼，來自 PyTorch 官方教學。
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 50):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.permute(1, 0, 2) # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        """
        # x (N, S, E) + pe (1, S, E) -> (N, S, E)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerAutoencoder(nn.Module):
    """
    使用 Transformer 作為 Autoencoder。
    - Input: (N, 1, 41, 20)
    - Reshaped: (N, 41, 20) [BatchSize, SeqLen, Features]
    - d_model: Transformer 內部的隱藏維度
    - latent_dim: 最終壓縮的潛在維度
    """
    def __init__(self, 
                 input_dim: int = 20, 
                 seq_len: int = 41,
                 latent_dim: int = 10, 
                #  d_model: int = 64, 
                #  nhead: int = 4, 
                #  num_encoder_layers: int = 3, 
                #  num_decoder_layers: int = 3, 
                #  dim_feedforward: int = 128, 
                 d_model: int = 128,            # Optuna 找到的值
                 nhead: int = 8,                # Optuna 找到的值
                 num_encoder_layers: int = 4,   # Optuna 找到的值
                 num_decoder_layers: int = 4,   # 保持與 encoder 對稱
                 dim_feedforward: int = 512,    # Optuna 找到的值
                 dropout: float = 0.1):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        
        # 1. Encoder
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=seq_len)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True # 確保 (N, S, E)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # 2. Bottleneck (Projection)
        self.encoder_proj = nn.Linear(d_model, latent_dim)
        self.decoder_proj = nn.Linear(latent_dim, d_model)

        # 3. Decoder
        decoder_layer = nn.TransformerEncoderLayer( # 我們使用 EncoderLayer 作為 Decoder
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_decoder_layers)
        
        self.output_proj = nn.Linear(d_model, input_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 1, 41, 20)
        x_in = x.squeeze(1) # (N, 41, 20)
        
        x_proj = F.relu(self.input_proj(x_in)) # (N, 41, d_model)
        x_pe = self.pos_encoder(x_proj)
        
        enc_out = self.transformer_encoder(x_pe) # (N, 41, d_model)
        
        # 使用 mean pooling
        enc_mean = enc_out.mean(dim=1) # (N, d_model)
        z = self.encoder_proj(enc_mean) # (N, latent_dim)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # z: (N, latent_dim)
        dec_in_proj = F.relu(self.decoder_proj(z)) # (N, d_model)
        
        # 廣播 latent vector 到所有 41 個位置
        dec_in_bcast = dec_in_proj.unsqueeze(1).repeat(1, self.seq_len, 1) # (N, 41, d_model)
        dec_in_pe = self.pos_encoder(dec_in_bcast)
        
        dec_out = self.transformer_decoder(dec_in_pe) # (N, 41, d_model)
        
        x_recon_flat = self.output_proj(dec_out) # (N, 41, input_dim)
        x_recon = x_recon_flat.unsqueeze(1) # (N, 1, 41, 20)
        return x_recon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        用於 autoencoder 訓練，只返回重建
        """
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon


class PricerTransformer(nn.Module):
    """
    Pricer 模型，使用 *固定* 的 TransformerAutoencoder 進行降維。
    """
    def __init__(self, 
                latent_dim: int, 
                pricing_param_dim: int, 
                tae_model_path: str,
                spot_param_dim: int = 1, # <--- 1. 新增 S0 維度
                # TAE 模型的架構參數 (必須與訓練時完全一致)
                input_dim: int = 20, 
                seq_len: int = 41,
                d_model: int = 64, 
                nhead: int = 4, 
                num_encoder_layers: int = 3, 
                num_decoder_layers: int = 3, 
                dim_feedforward: int = 128
                ):
        super().__init__()
        self.latent_dim = latent_dim
        self.pricing_param_dim = pricing_param_dim
        self.spot_param_dim = spot_param_dim # <--- 2. 儲存 S0 維度

        # 1. 加載預訓練的 TransformerAutoencoder (TAE)
        self.tae_model = TransformerAutoencoder(
            input_dim=input_dim,
            seq_len=seq_len,
            latent_dim=latent_dim,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward
        )

        # 加載狀態字典
        try:
            self.tae_model.load_state_dict(torch.load(tae_model_path))
            print(f"Loaded pre-trained TAE model from {tae_model_path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"TAE model file not found at {tae_model_path}")
        except RuntimeError as e:
            print(f"Error loading TAE state_dict. Did architecture change? {e}")
            raise

        # 2. **凍結 TAE 的參數**
        self.tae_model.eval()
        for param in self.tae_model.parameters():
            param.requires_grad = False

        # 3. Pricer MLP (與 PricerPCA/LDA 相同)
        self.pricing_mlp = nn.Sequential(
            nn.Linear(latent_dim + pricing_param_dim + spot_param_dim, 16), # <--- 3. 新的輸入維度
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, vol_surface: torch.Tensor, pricing_params: torch.Tensor, spot_params: torch.Tensor) -> torch.Tensor: # <--- 4. 接收 S0
        # 1. 使用 TAE Encoder 轉換波動率曲面
        # 在 no_grad 上下文中執行，因為 TAE 已被凍結
        with torch.no_grad():
            z = self.tae_model.encode(vol_surface) 

        # 2. 串聯潛在向量和定價參數
        mlp_input = torch.cat([z, pricing_params, spot_params], dim=1) # <--- 5. 串聯 S0

        # 3. 通過 MLP 進行定價
        price_pred = self.pricing_mlp(mlp_input)

        return price_pred
        
# --------------------------
# Training Functions (TAE)
# --------------------------

def train_and_save_TAE(
    folder: str,
    latent_dim: int = 10,
    batch_size: int = 32,
    num_epochs: int = 100, # TAE 可能需要更多/更少
    lr: float = 1e-4, # Transformer 對學習率敏感
    save_path: str = "tae_model.pt"
):
    """
    訓練一個 TransformerAutoencoder (TAE) 並保存。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加載數據
    train_loader, train_dataset = create_dataloader(
        folder, "post_vol_", "train", 
        batch_size=batch_size, 
        shuffle=True,  
        compute_stats=True 
    )
    test_loader, test_dataset = create_dataloader(
        folder, "post_vol_", "test", 
        batch_size=batch_size, 
        shuffle=False, 
        compute_stats=False
    )

    # 獲取形狀
    first_batch, _, _ = next(iter(train_loader))
    # (N, 1, 41, 20)
    seq_len = first_batch.shape[2] # 41
    input_dim = first_batch.shape[3] # 20

    # 初始化 TAE 模型
    model = TransformerAutoencoder(
        input_dim=input_dim,
        seq_len=seq_len,
        latent_dim=latent_dim
        # 使用默認的 d_model=64, nhead=4 等
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4) # AdamW 通常對 Transformer 更好
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr * 0.01)

    train_losses, test_losses = [], []
    
    print(f"--- 開始訓練 Transformer Autoencoder (TAE) ---")
    print(f"SeqLen={seq_len}, InputDim={input_dim} -> LatentDim={latent_dim}")
    print(f"d_model={model.d_model}, nhead={model.transformer_encoder.layers[0].self_attn.num_heads}")

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        for x, _, _ in train_loader:
            x = x.to(device)
            
            optimizer.zero_grad()
            x_recon = model(x)
            
            loss = F.mse_loss(x_recon, x) # 簡單的重建損失
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # 梯度裁剪
            optimizer.step()
            
            total_train_loss += loss.item() * x.size(0)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        avg_train_loss = total_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        # 評估測試集
        model.eval()
        total_test_loss = 0.0
        with torch.no_grad():
            for x_test, _, _ in test_loader:
                x_test = x_test.to(device)
                x_recon_test = model(x_test)
                loss_test = F.mse_loss(x_recon_test, x_test)
                total_test_loss += loss_test.item() * x_test.size(0)

        avg_test_loss = total_test_loss / len(test_loader.dataset)
        test_losses.append(avg_test_loss)

        print(f"Epoch {epoch+1}/{num_epochs} " f"| LR={current_lr:.2e} " f"| train_loss={avg_train_loss:.6f} " f"| test_loss={avg_test_loss:.6f}")

    # 保存模型 (state_dict)
    torch.save(model.state_dict(), save_path)
    print(f"TAE model state saved to {save_path}")

    # 保存損失歷史
    np.save(os.path.join(folder, "tae_train_losses.npy"), np.array(train_losses))
    np.save(os.path.join(folder, "tae_test_losses.npy"), np.array(test_losses))

    return model

def plot_tae_loss_curves(folder: str):
    """
    繪製 TAE 模型的訓練/測試損失曲線。
    """
    train_file = os.path.join(folder, "tae_train_losses.npy")
    test_file = os.path.join(folder, "tae_test_losses.npy")
    
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        print("Warning: TAE loss files not found. Skipping plot.")
        return

    train_losses = np.load(train_file)
    test_losses = np.load(test_file)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    epochs = np.arange(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="train", linewidth=2, color="purple")
    ax.plot(epochs, test_losses, label="test", linewidth=2, linestyle="--", color="plum")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reconstruction Loss (MSE)")
    ax.set_title(f"Transformer Autoencoder (TAE) Loss Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{folder}/tae_loss_curve.png", dpi=300)
    plt.show()
    plt.close()


def train_and_save_pricer_TAE(
    folder: str,
    product_type: str,
    tae_model_path: str, # 指向 .pt
    latent_dim: int = 10,
    pricing_param_dim: int = 2,
    batch_size: int = 128,
    num_epochs: int = 150, 
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> tuple:
    """
    訓練一個使用預訓練 TAE 模型的 PricerTransformer。
    (此函數與 PCA/LDA 版本幾乎相同)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(folder, exist_ok=True)

    # 加載定價數據
    train_loader, train_dataset = create_pricing_dataloader(folder, product_type, "train", batch_size=batch_size, shuffle=True, compute_stats=True)
    test_loader, test_dataset = create_pricing_dataloader(folder, product_type, "test", batch_size=batch_size, shuffle=False, compute_stats=False)

    # 創建 PricerTransformer 模型
    # 模型在初始化時會自動加載 tae_model_path
    model = PricerTransformer(
        latent_dim=latent_dim, 
        pricing_param_dim=pricing_param_dim, 
        spot_param_dim=1, # <--- 1. 傳入 S0 維度
        tae_model_path=tae_model_path
        # 使用默認的 TAE 架構參數
    ).to(device)

    # 優化器和調度器 (只優化 MLP 參數)
    optimizer = optim.Adam(model.pricing_mlp.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr * 0.01)

    train_losses, test_losses = [], []

    print("--- 開始訓練 Pricer (TAE) ---")

    for epoch in range(num_epochs):
        model.train() # 只會啟用 MLP 的 train mode
        total_train_loss = 0.0

        for vol_surface, pricing_params, spot_params, target_prices in train_loader: # <--- 2. 解開 4 個項目
            vol_surface = vol_surface.to(device)
            pricing_params = pricing_params.to(device)
            spot_params = spot_params.to(device) # <--- 3. S0 傳到 device
            target_prices = target_prices.to(device)

            # Forward pass
            predicted_prices = model(vol_surface, pricing_params, spot_params) # <--- 4. 傳入 S0

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
            for vol_surface_test, pricing_params_test, spot_params_test, target_prices_test in test_loader: # <--- 5. 解開 4 個項目
                vol_surface_test = vol_surface_test.to(device)
                pricing_params_test = pricing_params_test.to(device)
                spot_params_test = spot_params_test.to(device) # <--- 6. S0 傳到 device
                target_prices_test = target_prices_test.to(device)

                predicted_prices_test = model(vol_surface_test, pricing_params_test, spot_params_test) # <--- 7. 傳入 S0
                loss_test = F.mse_loss(predicted_prices_test, target_prices_test)
                total_test_loss += loss_test.item() * vol_surface_test.size(0)

        avg_test_loss = total_test_loss / len(test_loader.dataset)
        test_losses.append(avg_test_loss)

        print(f"Epoch {epoch+1}/{num_epochs} " f"| LR={current_lr:.2e} " f"| train_loss={avg_train_loss:.6f} " f"| test_loss={avg_test_loss:.6f}")

    # 保存模型狀態 (只保存 MLP 的權重) 和損失歷史
    state_path = os.path.join(folder, f"{product_type}_pricer_tae_state_dict.pt")
    torch.save(model.state_dict(), state_path)

    np.save(os.path.join(folder, f"{product_type}_pricer_tae_train_losses.npy"), np.array(train_losses))
    np.save(os.path.join(folder, f"{product_type}_pricer_tae_test_losses.npy"), np.array(test_losses))

    print(f"Saved pricer (TAE) model state to {state_path}")
    print(f"Saved pricer (TAE) train/test losses to {folder}")

    return model, train_losses, test_losses

# --------------------------
# Plotting Functions (TAE)
# --------------------------

def plot_loss_curves(folder: str, product_type: str, model_type: str = "TAE"):
    """
    繪製 Pricer 模型的訓練/測試損失曲線。
    model_type: "PCA", "VAE", "LDA", or "TAE"
    """
    
    if model_type.upper() == "TAE":
        pricer_train_file = os.path.join(folder, f"{product_type}_pricer_tae_train_losses.npy")
        pricer_test_file = os.path.join(folder, f"{product_type}_pricer_tae_test_losses.npy")
        title_prefix = "Pricer (TAE)"
    elif model_type.upper() == "PCA":
        # ... (PCA case) ...
        pricer_train_file = os.path.join(folder, f"{product_type}_pricer_pca_train_losses.npy")
        pricer_test_file = os.path.join(folder, f"{product_type}_pricer_pca_test_losses.npy")
        title_prefix = "Pricer (PCA)"
    elif model_type.upper() == "LDA":
        # ... (LDA case) ...
        pricer_train_file = os.path.join(folder, f"{product_type}_pricer_lda_train_losses.npy")
        pricer_test_file = os.path.join(folder, f"{product_type}_pricer_lda_test_losses.npy")
        title_prefix = "Pricer (LDA)"
    elif model_type.upper() == "VAE":
         # ... (VAE case) ...
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
    ax.plot(pricer_epochs, pricer_train_losses, label="train", linewidth=2, color="orange")
    ax.plot(pricer_epochs, pricer_test_losses, label="test", linewidth=2, linestyle="--", color="moccasin")
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


def visualize_latent_distribution_TAE(
    model_path: str, # .pt file
    folder: str, 
    latent_dim: int,
    save_path: str = None
):
    """
    可視化 TAE 潛在變量的分佈。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加載訓練數據
    train_loader, _ = create_dataloader(folder, "post_vol_", "train", batch_size=32, shuffle=False, compute_stats=False)
    
    # 加載 TAE 模型
    model = TransformerAutoencoder(latent_dim=latent_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 提取潛在表示
    latent_components = []
    with torch.no_grad():
        for x, _, _ in train_loader:
            x = x.to(device)
            z = model.encode(x) # (batch_size, latent_dim)
            latent_components.append(z.cpu().numpy())
            
    latent_components = np.concatenate(latent_components, axis=0)
    
    # --- 可視化 ---
    num_dims_to_plot = min(10, latent_dim)
    # ... (與 PCA/LDA 版本相同的繪圖代碼) ...
    num_rows = int(np.ceil(num_dims_to_plot / 2))
    fig, axes = plt.subplots(num_rows, 2, figsize=(12, num_rows * 3))
    axes = axes.flatten()

    for i in range(num_dims_to_plot):
        ax = axes[i]
        ax.hist(latent_components[:, i], bins=50, alpha=0.7, label=f"Z {i+1}", color="purple")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Distribution of TAE Latent Dim {i+1}")
        ax.legend()

    for i in range(num_dims_to_plot, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"TAE latent distribution saved to {save_path}")
    plt.show()

    return latent_components


def show_quote_date_reconstructions_TAE(
    folder: str,
    quote_dates: list,
    model_path: str, # .pt file
    latent_dim: int,
    device: Optional[Union[str, torch.device]] = None,
    cmap: str = "rainbow",
    data_type: str = "train",
):
    """
    顯示特定日期的 TAE 重建效果。
    """
    if len(quote_dates) != 3:
        raise ValueError(f"Expected exactly 3 quote dates, got {len(quote_dates)}")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加載 normalization stats
    vol_stats = np.load(f"{folder}/vol_normalization_stats.npz")
    vol_mean = vol_stats["mean"]
    vol_std = vol_stats["std"]
    print(f"Loaded normalization stats: mean={vol_mean:.6f}, std={vol_std:.6f}")

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
            
    # 加載 TAE 模型
    model = TransformerAutoencoder(latent_dim=latent_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

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
            recon = model(x) # (1, 1, H, W)

            x_np = x.squeeze(0).squeeze(0).cpu().numpy()
            recon_np = recon.squeeze(0).squeeze(0).cpu().numpy()

            # Denormalize
            x_denorm = x_np * vol_std + vol_mean
            recon_denorm = recon_np * vol_std + vol_mean
            figs.append((x_denorm, recon_denorm, actual_date, k_grid, T_grid))

    # --- 繪圖 (與 PCA 版本相同) ---
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    # ... (省略繪圖代碼，與 show_quote_date_reconstructions_PCA 相同) ...
    title_suffix = " (denormalized)"

    for idx, fig_data in enumerate(figs):
        x_display, recon_display, actual_date, k_grid, T_grid = fig_data
        T_mesh, k_mesh = np.meshgrid(T_grid.numpy(), k_grid.numpy())
        
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
        ax_out.set_title(f"TAE Recon: {actual_date}")
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
    plt.savefig(f"{folder}/tae_quote_date_reconstructions.png", dpi=300)
    plt.show()


def plot_predict_prices_from_vol_surface_and_params_TAE(
    folder: str,
    product_type: str,
    pricer_model_path: str, # .pt file for PricerTAE
    tae_model_path: str,    # .pt file for TAE
    include_train: bool = True,
    latent_dim: int = 10,
    pricing_param_dim: int = 2,
    batch_size: int = 32,
    device: Optional[Union[str, torch.device]] = None,
) -> dict:
    """
    使用訓練好的 PricerTAE 模型預測價格並繪圖。
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加載訓練好的 PricerTAE 模型
    # 我們必須同時加載 TAE 和 Pricer
    # 這裡我們必須加載 Pricer，它 *內部* 會加載 TAE
    model = PricerTransformer(
        latent_dim=latent_dim, 
        pricing_param_dim=pricing_param_dim, 
        spot_param_dim=1, # <--- 1. 傳入 S0 維度
        tae_model_path=tae_model_path
    )
    # 現在加載 Pricer (MLP) 的權重
    model.load_state_dict(torch.load(pricer_model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Loaded trained pricer (TAE) model from {pricer_model_path}")

    # 加載用於反標準化的統計數據 (與 PCA 版本相同)
    param_stats_path = f"{folder}/{product_type}_pricing_param_stats.npz"
    # ... (stats loading logic) ...
    stats = np.load(param_stats_path)
    price_params_mean = stats["params_mean"]
    price_params_std = stats["params_std"]
    price_mean = stats["price_mean"]
    price_std = stats["price_std"]

    def evaluate_dataset(data_type: str):
        # ... (Helper function (identical to PCA/LDA version)) ...
        print(f"\nLoading {product_type} {data_type} data for price prediction...")
        loader, dataset = create_pricing_dataloader(folder, product_type, data_type, batch_size=batch_size, shuffle=False, compute_stats=False)
        predicted_prices_list, target_prices_list = [], []

        print(f"Running predictions on {len(dataset)} {data_type} samples...")
        with torch.no_grad():
            for vol_surface, pricing_param, spot_param, target_price in loader: # <--- 2. 解開 4 個項目
                vol_surface = vol_surface.to(device)
                pricing_param = pricing_param.to(device)
                spot_param = spot_param.to(device) # <--- 3. S0 傳到 device
                target_price = target_price.to(device)
                predicted_price = model(vol_surface, pricing_param, spot_param) # <--- 4. 傳入 S0
                predicted_prices_list.append(predicted_price.cpu().numpy())
                target_prices_list.append(target_price.cpu().numpy())

        predicted_prices = np.concatenate(predicted_prices_list, axis=0)
        target_prices = np.concatenate(target_prices_list, axis=0)

        # Denormalize
        predicted_prices_denorm = predicted_prices * price_std + price_mean
        target_prices_denorm = target_prices * price_std + price_mean

        # Metrics
        mse = np.mean((predicted_prices_denorm - target_prices_denorm) ** 2)
        mae = np.mean(np.abs(predicted_prices_denorm - target_prices_denorm))
        rmse = np.sqrt(mse)
        ss_res = np.sum((target_prices_denorm - predicted_prices_denorm) ** 2)
        ss_tot = np.sum((target_prices_denorm - np.mean(target_prices_denorm)) ** 2)
        r2_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        print(f"\n{data_type.title()} Set Prediction Results (TAE Pricer):")
        print(f"R² Score: {r2_score:.6f}")
        print(f"MSE: {mse:.6f}, MAE: {mae:.6f}, RMSE: {rmse:.6f}")

        return {'predicted_prices': predicted_prices_denorm, 'target_prices': target_prices_denorm, 'r2_score': r2_score, 'mse': mse, 'mae': mae, 'rmse': rmse}


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
        # ... (Scatter plot logic, identical to PCA/LDA) ...
        result = results[data_type]
        predicted_flat = result['predicted_prices'].flatten()
        target_flat = result['target_prices'].flatten()

        ax_scatter = axes[i, 0]
        ax_scatter.scatter(target_flat, predicted_flat, alpha=0.6, s=20, label=f"{data_type} data")
        # ... (plot perfect line) ...
        min_price = min(np.min(target_flat), np.min(predicted_flat))
        max_price = max(np.max(target_flat), np.max(predicted_flat))
        ax_scatter.plot([min_price, max_price], [min_price, max_price], "r--", linewidth=2, label="Perfect Prediction")
        ax_scatter.set_xlabel("Ground Truth Price")
        ax_scatter.set_ylabel("Predicted Price")
        ax_scatter.set_title(f"{product_type} Price Prediction (TAE Pricer)\n{data_type.title()} Set (R² = {result['r2_score']:.4f})")
        ax_scatter.legend()
        ax_scatter.grid(True, alpha=0.3)
        stats_text = f"MSE: {result['mse']:.4f}\nMAE: {result['mae']:.4f}\nRMSE: {result['rmse']:.4f}"
        ax_scatter.text(0.05, 0.95, stats_text, transform=ax_scatter.transAxes, fontsize=10,
                        verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

        # ... (Residual plot logic, identical to PCA/LDA) ...
        ax_residual = axes[i, 1]
        residuals = predicted_flat - target_flat
        ax_residual.scatter(target_flat, residuals, alpha=0.6, s=20, color="green")
        ax_residual.axhline(y=0, color="r", linestyle="--", linewidth=2)
        ax_residual.set_xlabel("Ground Truth Price")
        ax_residual.set_ylabel("Prediction Error (Predicted - Truth)")
        ax_residual.set_title(f"{data_type.title()} Set Residuals")
        ax_residual.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{folder}/{product_type}_pricer_tae_prediction_comparison.png", dpi=300)
    plt.show()
    plt.close()

    return results