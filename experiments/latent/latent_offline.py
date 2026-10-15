# -*- coding: utf-8 -*-
"""
latent_offline.py
-----------------
只基于“已经训练好的 checkpoint”做 latent 分析：
- 不再训练模型
- 对每个 latent_size:
    * 从 ckpt 里读取模型参数（含 latents）
    * 在 test 集上算 NMSE
    * 在 train 集上算：
        (a) latent 之间的“差异度”（dissimilarity）
        (c) latent 与原始场之间的对应性（用相关矩阵的 MSE 表征）
- 在 outputs/latent/latent_offline/ 下保存
    * latent_offline_summary.png（三联图）
    * latent_offline_stats.npz（数值）

运行方式（和 main.py 类似）：
    python experiments/latent/latent_offline.py model=MMGNet datamodule.model_name=MMGN_NEW \
        datamodule.task=task1 datamodule.sampling_rate=[0.05,0.05]
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
import matplotlib.pyplot as plt

# ===================== 这里要你自己填 =====================
# 你已经训练好的 latent 维度列表
LATENT_LIST = [2, 4, 8, 16, 32, 64, 128, 256, 512]

# 对应每个 latent 维度的 best checkpoint 路径
CKPT_DICT = {
    2:   str(PROJECT_ROOT / "outputs" / "training" / "MMGN_task1_s_0.05/checkpoints/last.ckpt"),
    4:   str(PROJECT_ROOT / "outputs" / "training" / "MMGN_task1_s_0.05/checkpoints/last.ckpt"),
    8:   str(PROJECT_ROOT / "outputs" / "training" / "MMGN_task1_s_0.05/checkpoints/last.ckpt"),
    16:  str(PROJECT_ROOT / "outputs" / "training" / "MMGN_task1_s_0.05/checkpoints/last.ckpt"),
    32:  str(PROJECT_ROOT / "outputs" / "training" / "MMGN_task1_s_0.05/checkpoints/last.ckpt"),
    64:  str(PROJECT_ROOT / "outputs" / "training" / "MMGN_task1_s_0.05/checkpoints/last.ckpt"),
    128: str(PROJECT_ROOT / "outputs" / "training" / "MMGN_task1_s_0.05/checkpoints/last.ckpt"),
    256: str(PROJECT_ROOT / "outputs" / "training" / "MMGN_task1_s_0.05/checkpoints/last.ckpt"),
    512: str(PROJECT_ROOT / "outputs" / "training" / "MMGN_task1_s_0.05/checkpoints/last.ckpt"),
}
# ========================================================


# --------- 指标函数 ---------

def _rowwise_standardize(x: np.ndarray) -> np.ndarray:
    """对每一行做零均值 / 单位标准差标准化."""
    x_center = x - x.mean(axis=1, keepdims=True)
    std = x_center.std(axis=1, keepdims=True)
    std[std < 1e-8] = 1.0
    return x_center / std


def compute_latent_dissimilarity(latents: np.ndarray,
                                 max_samples: int = 256) -> float:
    """
    计算 latent code 之间的“差异度”。

    思路：把每个时间步的 latent 向量看成一个样本，
    计算样本之间的 Pearson 相关系数，然后取所有相关系数的标准差。
    latent: (N_code, L)
    返回值越大，表示 latent code 之间关系越“多样”（更不相似）。
    """
    if latents.ndim != 2:
        latents = latents.reshape(latents.shape[0], -1)

    N, L = latents.shape
    if N <= 1:
        return 0.0

    # 防止 N 太大，sub-sample 一部分时间步
    if N > max_samples:
        idx = np.linspace(0, N - 1, max_samples, dtype=int)
        latents = latents[idx]
        N = max_samples

    # 行标准化后，行与行之间的点积 / L 就是相关系数
    z = _rowwise_standardize(latents)  # (N, L)
    corr_mat = (z @ z.T) / float(L)    # (N, N)，理论范围 [-1, 1]
    corr_mat = np.clip(corr_mat, -1.0, 1.0)

    # 只取上三角（不含对角线）
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    vals = corr_mat[mask]
    if vals.size == 0:
        return 0.0
    return float(vals.std())


def compute_latent_data_mse(latents: np.ndarray,
                            temp: torch.Tensor,
                            mask3d: torch.Tensor,
                            max_steps: int = 256) -> float:
    """
    计算 latent 与原始场的“对应性”误差。

    思路：在时间维度上，把 latent 的时间序列和原始场的时间序列
    各自看成一个矩阵：
        Z  : (T, L)  —— 每一行一个时间步的 latent code
        Y  : (T, P)  —— 每一行一个时间步的场快照（只取海域像素展开后）

    分别对 Z 和 Y 做“样本为行”的标准化，然后计算：
        C_z = corr(Z)  (T x T)
        C_y = corr(Y)  (T x T)

    最终把两者的差的均方作为对应性误差，越小越好。
    """
    if latents.ndim != 2:
        latents = latents.reshape(latents.shape[0], -1)

    T_lat, L = latents.shape
    T_temp = temp.shape[0]
    T_mask = mask3d.shape[0]
    T = int(min(T_lat, T_temp, T_mask))
    if T <= 1:
        return 0.0

    # 时间步对齐
    latents = latents[:T]
    temp = temp[:T].contiguous()
    mask3d = mask3d[:T].contiguous()

    # 如有需要，等间隔采样一部分时间步，避免 T 过大
    if T > max_steps:
        idx = np.linspace(0, T - 1, max_steps, dtype=int)
        latents = latents[idx]
        temp = temp[idx]
        mask3d = mask3d[idx]
        T = max_steps

    # 原始场：只取海域像素
    temp_flat = temp.view(T, -1)          # (T, H*W)
    mask_flat = mask3d.view(T, -1)        # (T, H*W) 或 (T, H*W*C)
    if mask_flat.shape != temp_flat.shape:
        # 如果 mask 带有额外通道维，比如 (T,H,W,1)
        mask_flat = mask_flat.view(T, -1)
        min_len = min(mask_flat.shape[1], temp_flat.shape[1])
        temp_flat = temp_flat[:, :min_len]
        mask_flat = mask_flat[:, :min_len]

    # 取所有时间步都为海水的网格点，避免陆地区对相关性造成干扰
    valid_col = (mask_flat > 0.5).all(dim=0)   # (P,)
    if valid_col.sum() < 10:
        # 如果严格条件太苛刻，就退一步取“至少有一个时间步是海水”的位置
        valid_col = (mask_flat > 0.5).any(dim=0)
    temp_use = temp_flat[:, valid_col]        # (T, P_use)

    # 转 numpy
    Z = latents.astype(np.float64)           # (T, L)
    Y = temp_use.detach().cpu().numpy().astype(np.float64)  # (T, P_use)

    # 对每一行做标准化，然后算时间步之间的相关矩阵
    Z_std = _rowwise_standardize(Z)          # (T, L)
    Y_std = _rowwise_standardize(Y)          # (T, P_use)

    Cz = (Z_std @ Z_std.T) / float(L)
    Cy = (Y_std @ Y_std.T) / float(Y_std.shape[1])
    Cz = np.clip(Cz, -1.0, 1.0)
    Cy = np.clip(Cy, -1.0, 1.0)

    diff = Cz - Cy
    mse = float((diff ** 2).mean())
    return mse


def evaluate_nmse_on_test(lightning_module: pl.LightningModule,
                          datamodule: pl.LightningDataModule) -> float:
    """
    在 test_dataloader 上计算 NMSE：
        NMSE = sum( (yhat - y)^2 * mask ) / sum( y^2 * mask )
    只统计掩膜内像素。
    """
    device = next(lightning_module.parameters()).device
    lightning_module.eval()
    net = lightning_module.model  # 里面才是 MMGNet 网络

    num = 0.0
    den = 0.0
    with torch.no_grad():
        for batch in datamodule.test_dataloader():
            if len(batch) == 4:
                x, y, idx, mask = batch
            else:
                raise RuntimeError(
                    "test_dataloader 返回的 batch 不是 (x,y,idx,mask)，"
                    "当前脚本只支持 MMGN/MMGN_NEW 的 vizMMGN 数据集。"
                )
            x = x.to(device)
            y = y.to(device)
            idx = idx.to(device)
            mask = mask.to(device)

            yhat = net(x, idx)
            diff2 = ((yhat - y) * mask) ** 2
            num += diff2.sum().item()
            den += ((y * mask) ** 2).sum().item()

    nmse = num / (den + 1e-12)
    return float(nmse)


def compute_latent_stats(lightning_module: pl.LightningModule,
                         datamodule: pl.LightningDataModule):
    """
    综合计算：
      - latent 之间的 dissimilarity（std of pairwise corr）
      - latent 与原始场之间的对应性误差（相关矩阵 MSE）
    """
    net = getattr(lightning_module, "model", None)
    if net is None or not hasattr(net, "latents"):
        raise RuntimeError("当前模型没有 model.latents，本脚本只给 MMGN / MMGN_NEW 用。")

    # 1) latent matrix，形状大约是 (T_latent, L)
    latents = net.latents.detach().cpu().numpy()

    # 2) 训练集中的温度场和 mask，用于“原始数据”的相关性
    train_ds = datamodule.train_data
    temp = train_ds.temperature.float()
    mask3d = train_ds.mask_3d.float()

    dis = compute_latent_dissimilarity(latents)
    corr_mse = compute_latent_data_mse(latents, temp, mask3d)

    return dis, corr_mse


def save_results_and_plots(
        base_dir: str,
        latent_list: List[int],
        diss_runs: Dict[int, List[float]],
        nmse_runs: Dict[int, List[float]],
        corr_runs: Dict[int, List[float]]
):
    """
    保存 npz + 三联图
    """
    os.makedirs(base_dir, exist_ok=True)

    latent_arr = np.array(latent_list, dtype=int)
    diss_mean = np.array([np.mean(diss_runs[L]) for L in latent_list], dtype=float)
    corr_mean = np.array([np.mean(corr_runs[L]) for L in latent_list], dtype=float)
    nmse_mean = np.array([np.mean(nmse_runs[L]) for L in latent_list], dtype=float)

    np.savez(
        os.path.join(base_dir, "latent_offline_stats.npz"),
        latent_sizes=latent_arr,
        dissimilarity=diss_mean,
        correspondence=corr_mean,  # 这里存的是“相关矩阵 MSE”，数值越小越好
        nmse_mean=nmse_mean,
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # (a) dissimilarity
    ax = axes[0]
    ax.bar(range(len(latent_list)), diss_mean)
    ax.set_xticks(range(len(latent_list)))
    ax.set_xticklabels(latent_list)
    ax.set_xlabel("latent dimension L")
    ax.set_ylabel("std of pairwise corr")
    ax.set_title("(a) Dissimilarity of latent codes")
    for i, v in enumerate(diss_mean):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    # (b) NMSE
    ax = axes[1]
    box_data = [nmse_runs[L] for L in latent_list]
    ax.boxplot(
        box_data,
        positions=np.arange(1, len(latent_list) + 1),
        showmeans=True,
    )
    ax.set_xticks(np.arange(1, len(latent_list) + 1))
    ax.set_xticklabels(latent_list)
    ax.set_xlabel("latent dimension L")
    ax.set_ylabel("NMSE")
    ax.set_title("(b) NMSE on (viz) test set")

    # (c) correspondence error
    ax = axes[2]
    ax.bar(range(len(latent_list)), corr_mean)
    ax.set_xticks(range(len(latent_list)))
    ax.set_xticklabels(latent_list)
    ax.set_xlabel("latent dimension L")
    ax.set_ylabel("MSE between corr. matrices")
    ax.set_title("(c) Correspondence between latents and original data")
    for i, v in enumerate(corr_mean):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig_path = os.path.join(base_dir, "latent_offline_summary.png")
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)


# ----------------- 主函数：只读 ckpt 不训练 -----------------

@hydra.main(version_base="1.2", config_path="../../configs", config_name="default")
def main(cfg: DictConfig):
    model_name = cfg.datamodule.model_name
    if model_name not in ("MMGN", "MMGN_NEW"):
        raise ValueError(
            f"当前 datamodule.model_name = {model_name}，"
            f"latent_offline.py 目前只支持 MMGN / MMGN_NEW。"
        )

    # 使用项目内固定的 latent 输出目录，与启动位置无关。
    base_run_dir = PROJECT_ROOT / "outputs" / "latent"
    out_dir = os.path.join(base_run_dir, "latent_offline")
    os.makedirs(out_dir, exist_ok=True)

    diss_runs: Dict[int, List[float]] = defaultdict(list)
    nmse_runs: Dict[int, List[float]] = defaultdict(list)
    corr_runs: Dict[int, List[float]] = defaultdict(list)

    # 先构造 datamodule / normalizer，一次即可（数据和 latent 维度没关系）
    datamodule: pl.LightningDataModule = instantiate(cfg.datamodule)
    datamodule.prepare_data()
    normalizer = datamodule.setup()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for L in LATENT_LIST:
        if L not in CKPT_DICT:
            raise ValueError(f"CKPT_DICT 里没有 latent_size={L} 的路径，请补上。")

        ckpt_path = CKPT_DICT[L]
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"找不到 checkpoint: {ckpt_path}")

        print(f"\n========== [offline] latent_size = {L} ==========")
        print(f"  加载 ckpt: {ckpt_path}")

        # 修改 cfg 里的 latent_size（只是为了和模型结构一致）
        cfg.model.params_model.latent_size = int(L)

        # 实例化 LightningModule（不会训练）
        model: pl.LightningModule = instantiate(cfg.model, normalizer)

        # 从 ckpt 加载权重
        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt["state_dict"]
        model.load_state_dict(state_dict, strict=True)
        model.to(device)

        # 指标计算
        nmse = evaluate_nmse_on_test(model, datamodule)
        dis, corr_mse = compute_latent_stats(model, datamodule)

        print(f"[L={L}] NMSE = {nmse:.6e}, "
              f"dissimilarity(std corr) = {dis:.4f}, corr_MSE = {corr_mse:.4f}")

        nmse_runs[L].append(nmse)
        diss_runs[L].append(dis)
        corr_runs[L].append(corr_mse)

    save_results_and_plots(
        base_dir=out_dir,
        latent_list=LATENT_LIST,
        diss_runs=diss_runs,
        nmse_runs=nmse_runs,
        corr_runs=corr_runs,
    )
    print(f"\n全部完成，结果和图已保存在：{out_dir}")


if __name__ == "__main__":
    main()
