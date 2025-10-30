from ML_analyze import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os # 確保 import os

# *** 修正：刪除 unflatten_grid 函數，它是不必要的且會導致維度混淆 ***

def profile_likelihood_dim(s, eps=1e-12):
    """
    Automatic dimensionality selection via the profile-likelihood method
    of Zhu & Ghodsi (2006). (此函數不變)
    """
    s = np.asarray(s, dtype=float)
    p = s.size
    loglik = np.empty(p - 1)

    for q in range(1, p):  # candidate cut–off
        S1, S2 = s[:q], s[q:]  # two “groups” of values
        n1, n2 = q, p - q
        mu1, mu2 = S1.mean(), S2.mean()

        # pooled estimate of the common variance  σ²ˆ  (Eq. 8 in the paper)
        var1, var2 = S1.var(ddof=1), S2.var(ddof=1)
        sigma2 = max(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2), eps)

        # log-likelihood under a Gaussian model with common σ²  (Eqs. 1–2)
        ll = -0.5 * (n1 + n2) * np.log(2 * np.pi * sigma2)
        ll -= 0.5 * ((S1 - mu1) ** 2).sum() / sigma2
        ll -= 0.5 * ((S2 - mu2) ** 2).sum() / sigma2
        loglik[q - 1] = ll

    q_hat = np.argmax(loglik) + 1  # +1 because q starts at 1
    return q_hat, loglik

def svd_analysis(folder, all_w_grid, k_grid, T_grid):
    
    # all_w_grid shape is (N_samples, 41, 20)
    n_samples, H, W = all_w_grid.shape # H=41, W=20
    features_dim = H * W # 820

    all_w_grid_flatten = all_w_grid.reshape(all_w_grid.shape[0], -1)
    print("all_w_grid.shape", all_w_grid.shape)
    print("all_w_grid_flatten.shape", all_w_grid_flatten.shape)

    # Perform SVD
    # Vt shape will be (k, 820)
    U, s, Vt = np.linalg.svd(all_w_grid_flatten, full_matrices=False)
    q_hat, loglik = profile_likelihood_dim(s)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    # Plot singular values
    idx = np.arange(len(s))
    ax1.plot(idx, s, marker="o", label="Singular value", color="tab:blue")
    
    # *** 建議：使用 log scale 來匹配論文中的 Scree Plot ***
    ax1.set_yscale("log")
    ax1.set_xlabel("Index")
    ax1.set_ylabel("Singular Value", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5)

    # right-hand y-axis: profile log-likelihood
    ax2 = ax1.twinx()
    # *** 修正：限制繪圖範圍，使其更易讀 ***
    plot_range = min(len(s) - 1, 100) # 只看前 100 個維度
    ax2.plot(idx[1:plot_range+1], loglik[:plot_range], marker="x", label="Profile log-likelihood", color="tab:red")
    ax2.set_ylabel("Profile Log-Likelihood", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    # mark optimal dimensionality
    ax2.axvline(q_hat, color="tab:green", linestyle="--", linewidth=1, label=f"$\\hat{{q}}={q_hat}$ (Optimal Dim)")

    # combine legends from both axes
    lines = ax1.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="best")

    plt.title("SVD Scree Plot with Profile Log-Likelihood")
    plt.tight_layout()
    plt.savefig(f"{folder}/svd_singular_values_and_profile.png")
    plt.show()

    # --- Plot the first nine singular vectors as heatmaps ---
    plt.figure(figsize=(15, 12))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        # Vt contains the right singular vectors (modes of variation in volatility surfaces)
        vec = Vt[i] # Shape (820,)
        # Reshape the vector back to the 2D grid
        vec_2d = vec.reshape(H, W) # Shape (41, 20)

        # *** 修正：imshow 的 extent 參數 ***
        # extent=[col_min, col_max, row_min, row_max]
        # col (X-axis) 是 T (20), row (Y-axis) 是 k (41)
        extent = [min(T_grid), max(T_grid), min(k_grid), max(k_grid)]
        img = plt.imshow(vec_2d, aspect="auto", cmap="viridis", origin="lower", extent=extent)
        
        plt.colorbar(img, label=f"SV {i+1} Value")
        plt.title(f"Singular Vector {i+1}")
        
        # *** 修正：xlabel 和 ylabel 應與 extent 對應 ***
        plt.ylabel("Strike (k)") # Y-axis
        plt.xlabel("Time to Maturity (T)") # X-axis

    plt.tight_layout()
    plt.savefig(f"{folder}/svd_singular_vectors.png")
    plt.show()

    # --- Randomly select a vol surface and show original vs. reconstruction ---
    idx = np.random.randint(0, all_w_grid.shape[0])
    sample_surf_2d = all_w_grid[idx] # Shape (41, 20)
    
    # *** 修正：這就是您遇到 ValueError 的地方 ***
    # 我們必須使用展平的 surface 來進行投影
    
    sample_surf_flat = sample_surf_2d.flatten() # Shape (820,)
    
    # 初始化一個展平的 projection
    projection_flat = np.zeros_like(sample_surf_flat) # Shape (820,)
    
    n_components_recon = 6
    for i in range(n_components_recon):
        # 1. 取得第 i 個主成分 (奇異向量)
        sv_i = Vt[i] # Shape (820,)
        # 2. 計算投影 (點積), 現在是 (820,) . (820,) -> scalar
        weight = np.dot(sample_surf_flat, sv_i)
        # 3. 疊加
        projection_flat += weight * sv_i # scalar * (820,)

    # *** 修正：將展平的結果重塑回 2D 以進行繪圖 ***
    projection_2d = projection_flat.reshape(H, W) # Shape (41, 20)
    residual_2d = sample_surf_2d - projection_2d # Shape (41, 20)

    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # *** 修正：移除對 unflatten_grid 的呼叫 ***
    # surf_2d, k_unique, T_unique = unflatten_grid(sample_surf, k_grid, T_grid) # <- 刪除
    # proj_2d, _, _ = unflatten_grid(projection, k_grid, T_grid) # <- 刪除
    # resid_2d, _, _ = unflatten_grid(residual, k_grid, T_grid) # <- 刪除

    # Common color scale for original and reconstruction
    vmin = min(sample_surf_2d.min(), projection_2d.min())
    vmax = max(sample_surf_2d.max(), projection_2d.max())
    
    # *** 修正：再次修正 extent 和 labels ***
    extent = [min(T_grid), max(T_grid), min(k_grid), max(k_grid)]

    # Original surface
    im0 = axes[0].imshow(sample_surf_2d, aspect="auto", cmap="viridis", origin="lower", extent=extent, vmin=vmin, vmax=vmax)
    axes[0].set_title(f"Original Vol Surface (Sample {idx})")
    axes[0].set_xlabel("Time to Maturity (T)")
    axes[0].set_ylabel("Strike (k)")
    fig.colorbar(im0, ax=axes[0])

    # Reconstruction using 6 singular vectors
    im1 = axes[1].imshow(projection_2d, aspect="auto", cmap="viridis", origin="lower",
                         extent=extent,
                         vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Reconstruction ({n_components_recon} SVs)")
    axes[1].set_xlabel("Time to Maturity (T)")
    fig.colorbar(im1, ax=axes[1])

    # Residual
    im2 = axes[2].imshow(residual_2d, aspect="auto", cmap="coolwarm", origin="lower",
                         extent=extent)
    axes[2].set_title("Residual (Original - Reconstruction)")
    axes[2].set_xlabel("Time to Maturity (T)")
    fig.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.savefig(f"{folder}/vol_surface_reconstruction.png")
    plt.show()

    return U, s, Vt