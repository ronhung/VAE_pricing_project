import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PCA_model import create_dataloader # 假設我們可以從您那裡導入

def load_all_surfaces(data_loader):
    """
    一個輔助函數，用來加載所有資料並展平。
    這與您在 PCAWrapper.fit 中做的事情相同。
    """
    all_surfaces = []
    print("Gathering all surfaces for K-Means...")
    # 確保 data_loader 的 shuffle=False
    for batch, _, _ in data_loader:
        all_surfaces.append(batch.numpy())
    
    surfaces_np = np.concatenate(all_surfaces, axis=0)
    # 展平為 (num_samples, H*W)
    surfaces_flat = surfaces_np.reshape(surfaces_np.shape[0], -1)
    print(f"Data shape for K-Means: {surfaces_flat.shape}")
    return surfaces_flat

def plot_kmeans_elbow_curve(folder, max_k=20):
    """
    繪製手肘法曲線來幫助決定 K
    """
    # 1. 加載數據
    # 只需要訓練數據，且 shuffle 應為 False 以便於重現
    train_loader, _ = create_dataloader(
        folder, "post_vol_", "train", 
        batch_size=128, 
        shuffle=False, 
        compute_stats=True 
    )
    surfaces_flat = load_all_surfaces(train_loader)
    
    # 2. 測試不同的 K
    inertias = []
    k_range = range(2, max_k + 1)
    
    print("Calculating inertias for different K values (Elbow Method)...")
    for k in k_range:
        kmeans = KMeans(
            n_clusters=k, 
            init='k-means++',  # 推薦的初始化方法
            n_init=10,         # 運行 10 次並取最好結果
            max_iter=300,
            random_state=42
        )
        kmeans.fit(surfaces_flat)
        inertias.append(kmeans.inertia_)
        print(f"  K={k}, Inertia={kmeans.inertia_:.2f}")

    # 3. 繪圖
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertias, 'bo-')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia (WCSS)')
    plt.title('K-Means Elbow Method')
    plt.xticks(k_range)
    plt.grid(True)
    plt.savefig(f"{folder}/kmeans_elbow_curve.png")
    plt.show()
    print(f"Elbow curve saved to {folder}/kmeans_elbow_curve.png")

# --- 如何執行 ---
if __name__ == "__main__":
    folder = "../data_process/data_pack"
    plot_kmeans_elbow_curve(folder, max_k=20)