
"""
latent_scatter.py  (t-SNE 版本，无 sklearn/scipy 依赖)
----------------------------------------------------
按论文思路，用 t-SNE 在同一个 2D 平面中可视化
不同 latent_size 模型的 latent 变量相似性。

步骤：
1. 从每个 checkpoint 中读取 model.latents (T, L)
2. 对每个模型，转置为 (L, T)，每一行是一个 latent variable
3. 把所有模型的 latent variables 堆成大矩阵 Z (sum_L L, T)
4. 用纯 numpy 实现的 t-SNE 在 Z 上跑一次，得到 Z_emb (sum_L L, 2)
5. 按 latent size 划分回去：
    - 绿色：latent_size = 128 的 latent variables（基准）
    - 红色：其他 latent_size = L 的 latent variables
6. 生成 3×3 总图 + 每个 L 的单独子图

依赖：
    numpy, torch, matplotlib
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
CKPT_DICT = {
    1:   str(PROJECT_ROOT / "outputs" / "latent" / "latent_sweep/latent_L1_seed3407/last.ckpt"),
    2:   str(PROJECT_ROOT / "outputs" / "latent" / "latent_sweep/latent_L2_seed3407/last.ckpt"),
    4:   str(PROJECT_ROOT / "outputs" / "latent" / "latent_sweep/latent_L4_seed3407/last.ckpt"),
    8:   str(PROJECT_ROOT / "outputs" / "latent" / "latent_sweep/latent_L8_seed3407/last.ckpt"),
    16:  str(PROJECT_ROOT / "outputs" / "latent" / "latent_sweep/latent_L16_seed3407/last.ckpt"),
    32:  str(PROJECT_ROOT / "outputs" / "latent" / "latent_sweep/latent_L32_seed3407/last.ckpt"),
    64:  str(PROJECT_ROOT / "outputs" / "latent" / "latent_sweep/latent_L64_seed3407/last.ckpt"),
    128: str(PROJECT_ROOT / "outputs" / "latent" / "latent_sweep/latent_L128_seed3407/last.ckpt"),
    256: str(PROJECT_ROOT / "outputs" / "latent" / "latent_sweep/latent_L256_seed3407/last.ckpt"),
    512: str(PROJECT_ROOT / "outputs" / "latent" / "latent_sweep/latent_L512_seed3407/last.ckpt"),
}

BASE_LATENT_DIM = 128

# t-SNE
TSNE_PERPLEXITY = 30.0
TSNE_SEED = 3407
TSNE_N_ITER = 1000
TSNE_LR = 200.0

# 为了图像不太密，每组最多画多少个点（latent variables）
MAX_POINTS_PER_GROUP = 128

# 输出目录
OUT_DIR = PROJECT_ROOT / "outputs" / "latent" / "latent_tsne_plots"


# ----------------------------------------------------
# 下面是纯 numpy 实现的 t-SNE
# ----------------------------------------------------
def _pairwise_distances(X: np.ndarray) -> np.ndarray:
    """X: (N, D) -> 欧氏距离平方矩阵 (N, N)"""
    sum_X = np.sum(np.square(X), axis=1)
    dists = sum_X[:, None] + sum_X[None, :] - 2.0 * np.dot(X, X.T)
    dists = np.maximum(dists, 0.0)
    return dists


def _Hbeta(Di: np.ndarray, beta: float):
    """给定某个点到其它点的距离 Di 和 beta=1/(2*sigma^2)，计算熵 H 和 P"""
    P = np.exp(-Di * beta)
    P[Di == 0] = 0.0  # 自身置 0
    sumP = np.sum(P)
    if sumP <= 0.0:
        return 0.0, np.zeros_like(Di)
    P = P / sumP
    H = -np.sum(P * np.log2(P + 1e-10))
    return H, P


def _binary_search_perplexity(D: np.ndarray,
                              perplexity: float,
                              tol: float = 1e-5,
                              max_iter: int = 50):
    """对每个点二分搜索合适的 beta 使得感知度接近期望 perplexity"""
    N = D.shape[0]
    betas = np.ones((N, 1), dtype=np.float64)
    logU = np.log2(perplexity)
    P = np.zeros((N, N), dtype=np.float64)

    for i in range(N):
        betamin = -np.inf
        betamax = np.inf

        # 去掉对角线
        Di = np.concatenate((D[i, :i], D[i, i + 1:]))

        H, thisP = _Hbeta(Di, betas[i])
        Hdiff = H - logU
        tries = 0

        while np.abs(Hdiff) > tol and tries < max_iter:
            if Hdiff > 0:
                betamin = betas[i].copy()
                if np.isinf(betamax):
                    betas[i] *= 2.0
                else:
                    betas[i] = (betas[i] + betamax) / 2.0
            else:
                betamax = betas[i].copy()
                if np.isinf(betamin):
                    betas[i] /= 2.0
                else:
                    betas[i] = (betas[i] + betamin) / 2.0

            H, thisP = _Hbeta(Di, betas[i])
            Hdiff = H - logU
            tries += 1

        # 把 thisP 填回 P 的第 i 行
        P[i, :i] = thisP[:i]
        P[i, i + 1:] = thisP[i:]

    return P


def tsne_numpy(X: np.ndarray,
               n_components: int = 2,
               perplexity: float = 30.0,
               n_iter: int = 1000,
               learning_rate: float = 200.0,
               random_state: int = 0) -> np.ndarray:
    """
    纯 numpy 版 t-SNE。
    X: (N, D)，返回 Y: (N, n_components)
    """
    np.random.seed(random_state)
    X = X.astype(np.float64)
    N, D = X.shape

    # 1) pairwise distances
    Dmat = _pairwise_distances(X)

    # 2) 高维相似度 P
    P = _binary_search_perplexity(Dmat, perplexity=perplexity)
    P = (P + P.T) / (2.0 * N)  # 对称化并缩放
    P = np.maximum(P, 1e-12)

    # 3) 初始化低维表示
    Y = np.random.normal(0.0, 1e-4, size=(N, n_components))
    dY = np.zeros_like(Y)
    iY = np.zeros_like(Y)
    gains = np.ones_like(Y)

    momentum = 0.5
    final_momentum = 0.8
    switch_iter = 250

    for it in range(n_iter):
        # 低维 Student-t 分布相似度 Q
        sum_Y = np.sum(np.square(Y), axis=1)
        num = 1.0 / (1.0 + sum_Y[:, None] + sum_Y[None, :] - 2.0 * np.dot(Y, Y.T))
        np.fill_diagonal(num, 0.0)
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        # 梯度
        PQ = P - Q
        for i in range(N):
            dY[i, :] = np.sum((PQ[:, i] * num[:, i])[:, None] * (Y[i, :] - Y), axis=0)

        # 自适应步长
        gains = (gains + 0.2) * ((dY > 0) != (iY > 0)) + \
                (gains * 0.8) * ((dY > 0) == (iY > 0))
        gains[gains < 0.01] = 0.01

        # 更新
        iY = momentum * iY - learning_rate * (gains * dY)
        Y = Y + iY
        Y = Y - Y.mean(axis=0, keepdims=True)

        if it == switch_iter:
            momentum = final_momentum

        if (it + 1) % 100 == 0:
            C = np.sum(P * np.log(P / Q))
            print(f"Iter {it+1}/{n_iter}, KL divergence = {C:.4f}")

    return Y.astype(np.float32)


# ----------------------------------------------------
# 读 ckpt，取出 model.latents (T, L)
# ----------------------------------------------------
def load_latents_from_ckpt(ckpt_path: str) -> np.ndarray:
    """
    从 Lightning ckpt 中找到 'latents' 参数，返回 shape=(T, L) 的 numpy 数组。
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)

    lat_key = None
    for k in state.keys():
        if k.endswith(".latents") or k == "latents":
            lat_key = k
            break

    if lat_key is None:
        keys_preview = list(state.keys())[:20]
        raise RuntimeError(
            f"在 {ckpt_path} 中没找到 'latents' 参数，"
            f"state_dict 前 20 个 key: {keys_preview}"
        )

    latents = state[lat_key]     # Tensor (T, L)
    latents = latents.detach().cpu().numpy()
    print(f"  -> found latents: key='{lat_key}', shape={latents.shape}")
    return latents


# ----------------------------------------------------
# 主流程：按论文思路构造 Z，并用 t-SNE 可视化
# ----------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) 加载所有模型的 latents (T, L)
    latents_time_first = {}  # L -> (T, L)
    T_set = set()

    for L, path in CKPT_DICT.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"latent_size={L} 的 checkpoint 不存在: {path}"
            )
        print(f"加载 latent_size = {L} 的 checkpoint: {path}")
        lat = load_latents_from_ckpt(path)   # (T, L)
        latents_time_first[L] = lat
        T_set.add(lat.shape[0])

    if len(T_set) != 1:
        raise RuntimeError(f"不同模型的时间步数不一致: {T_set}")
    T = list(T_set)[0]
    print(f"\n所有模型的时间步数 T = {T}")

    if BASE_LATENT_DIM not in latents_time_first:
        raise ValueError(f"BASE_LATENT_DIM={BASE_LATENT_DIM} 不在 CKPT_DICT 中。")

    # 2) 构造大矩阵 Z: 每一行是一个 latent variable (L, T)
    latent_dims_all = sorted(latents_time_first.keys())  # 包含 128
    all_latent_vars = []
    index_map = {}   # L -> (start, end)

    start = 0
    for L in latent_dims_all:
        lat = latents_time_first[L]   # (T, L)
        vars_L = lat.T               # (L, T)
        all_latent_vars.append(vars_L)
        end = start + vars_L.shape[0]
        index_map[L] = (start, end)   # [start, end)
        start = end

    Z = np.concatenate(all_latent_vars, axis=0)  # (sum_L L, T)
    N_total = Z.shape[0]
    print(f"大矩阵 Z 形状: {Z.shape} (总 latent variables: {N_total})")

    # 3) 在 Z 上跑一次 t-SNE
    print("\n开始运行 t-SNE（纯 numpy 实现）...")
    Z_emb = tsne_numpy(
        Z,
        n_components=2,
        perplexity=TSNE_PERPLEXITY,
        n_iter=TSNE_N_ITER,
        learning_rate=TSNE_LR,
        random_state=TSNE_SEED,
    )
    print("t-SNE 完成。")

    # 4) 按 latent size 划分回去
    coords_by_L = {}
    for L in latent_dims_all:
        st, ed = index_map[L]
        coords_by_L[L] = Z_emb[st:ed]   # (L, 2)

    base_coords_full = coords_by_L[BASE_LATENT_DIM]

    # 需要绘制子图的 latent 维度列表：去掉 128，只画 1,2,4,8,16,32,64,256,512
    plot_latent_list = sorted([L for L in latent_dims_all if L != BASE_LATENT_DIM])

    # 小工具：限制每组最多 MAX_POINTS_PER_GROUP 个点
    rng = np.random.default_rng(TSNE_SEED)

    def sample_points(arr: np.ndarray, max_points: int) -> np.ndarray:
        n = arr.shape[0]
        if n <= max_points:
            return arr
        idx = rng.choice(n, size=max_points, replace=False)
        return arr[idx]

    # 5) 画 3×3 汇总图
    fig_grid, axes = plt.subplots(3, 3, figsize=(10, 8))

    for ax, L in zip(axes.flat, plot_latent_list):
        base_plot = sample_points(base_coords_full, MAX_POINTS_PER_GROUP)
        coords_L = coords_by_L[L]
        coords_L_plot = sample_points(coords_L, MAX_POINTS_PER_GROUP)

        ax.scatter(
            base_plot[:, 0], base_plot[:, 1],
            s=35, alpha=0.35,
            color="tab:green", edgecolors="none",
            label=str(BASE_LATENT_DIM),
        )
        ax.scatter(
            coords_L_plot[:, 0], coords_L_plot[:, 1],
            s=35, alpha=0.85,
            color="tab:red", edgecolors="none",
            label=str(L),
        )
        ax.set_xticks([])
        ax.set_yticks([])

        leg = ax.legend(loc="lower right", fontsize=8, frameon=True)
        leg.get_frame().set_alpha(0.8)

        # 同时保存单独一张图
        fig_single, ax_single = plt.subplots(figsize=(4, 4))
        ax_single.scatter(
            base_plot[:, 0], base_plot[:, 1],
            s=35, alpha=0.35,
            color="tab:green", edgecolors="none",
            label=str(BASE_LATENT_DIM),
        )
        ax_single.scatter(
            coords_L_plot[:, 0], coords_L_plot[:, 1],
            s=35, alpha=0.85,
            color="tab:red", edgecolors="none",
            label=str(L),
        )
        ax_single.set_xticks([])
        ax_single.set_yticks([])
        leg2 = ax_single.legend(loc="lower right", fontsize=8, frameon=True)
        leg2.get_frame().set_alpha(0.8)

        single_path = os.path.join(OUT_DIR, f"latent_tsne_L{L}.png")
        fig_single.tight_layout()
        fig_single.savefig(single_path, dpi=300)
        plt.close(fig_single)
        print(f"  已保存单图: {single_path}")

    fig_grid.tight_layout()
    grid_path = os.path.join(OUT_DIR, "latent_tsne_grid.png")
    fig_grid.savefig(grid_path, dpi=300)
    plt.close(fig_grid)

    print(f"\n已保存 3×3 汇总图: {grid_path}")
    print(f"所有输出都在目录: {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()
