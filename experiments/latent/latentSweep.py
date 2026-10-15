# -*- coding: utf-8 -*-
"""
latent_sweep.py
---------------
针对 MMGN / MMGN_NEW 模型，扫一组 latent_size：
- 每个 latent_size 重新训练一次
- 计算：
  (a) latent code 之间的差异度 dissimilarity
  (b) 测试集 NMSE（只在海域掩膜内统计）
  (c) latent 与样本平均场强度的相关性（对应性）
- 最后在 outputs/latent/latent_sweep/ 下生成三联图 latent_sweep_summary.png
  和结果 latent_sweep_stats.npz

运行方式示例（和 main.py 一样）：
    python experiments/latent/latentSweep.py model=MMGNet datamodule=current
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
import gc
from collections import defaultdict
from typing import List, Dict

import numpy as np
import torch
import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate

import matplotlib.pyplot as plt

# ===== 在这里改想要扫的 latent 维度 & 种子 =====
LATENT_LIST = [2, 4, 8, 16, 32, 64, 128, 256, 512]  # latent 维度列表
SEEDS = [3407]  # 想做 boxplot 就多放几个 seed


# ------------------------------------------------
# 一些小工具函数
# ------------------------------------------------
def compute_latent_dissimilarity(latents: np.ndarray,
                                 max_samples: int = 200) -> float:
    """
    计算 latent code 的“差异度”：
    这里用 1 - mean( pairwise cosine similarity )，越大表示越不相似
    latents: (N, L)
    """
    N, D = latents.shape
    if N > max_samples:
        idx = np.random.choice(N, max_samples, replace=False)
        latents = latents[idx]
        N = max_samples

    # 去掉每个样本自身的均值，避免仅靠偏移
    latents = latents - latents.mean(axis=1, keepdims=True)
    # 行归一化
    norm = np.linalg.norm(latents, axis=1, keepdims=True) + 1e-8
    z = latents / norm  # (N, L)
    # 余弦相似度矩阵
    G = z @ z.T  # (N, N)
    mask = ~np.eye(N, dtype=bool)
    mean_sim = G[mask].mean()
    dis = 1.0 - float(mean_sim)
    return dis


def compute_latent_data_corr(latents: np.ndarray,
                             feat: np.ndarray) -> float:
    """
    计算 latent code 与某个“样本级特征”之间的相关性。
    这里 feat 是每个样本的平均海域温度 (N,)。
    返回：所有 latent 维度绝对相关系数的平均值。
    """
    N, L = latents.shape
    assert feat.shape[0] == N

    corrs = []
    feat_center = feat - feat.mean()
    feat_std = feat_center.std() + 1e-8

    for j in range(L):
        lj = latents[:, j]
        lj_center = lj - lj.mean()
        lj_std = lj_center.std()
        if lj_std < 1e-8:
            corrs.append(0.0)
            continue
        cov = np.mean(lj_center * feat_center)
        corr = cov / (lj_std * feat_std)
        corrs.append(abs(float(corr)))

    return float(np.mean(corrs))


def evaluate_nmse_on_test(lightning_module: pl.LightningModule,
                          datamodule: pl.LightningDataModule) -> float:
    """
    在 test_dataloader 上计算 NMSE：
        NMSE = sum( (yhat - y)^2 * mask ) / sum( y^2 * mask )
    注意这里只统计掩膜内（海域）像素。
    只针对 MMGN / MMGN_NEW 的 vizMMGN 测试集 (x, y, idx, mask)。
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
    计算：
      - latent 之间的 dissimilarity
      - latent 与“平均海温”之间的相关性

    做法更鲁棒：
      1) 先取 N = latent 个数
      2) 从 train_data 中取 temperature 和 mask_3d
      3) 用 min(N, T_temp, T_mask) 对齐时间维，防止形状不一致
      4) 逐样本计算“掩膜内平均温度”作为 feat，再和 latent 做相关性
    """
    # 1) latent matrix
    net = getattr(lightning_module, "model", None)
    if net is None or not hasattr(net, "latents"):
        raise RuntimeError("当前模型没有 model.latents，本脚本只给 MMGN / MMGN_NEW 用。")

    latents = net.latents.detach().cpu().numpy()  # (N, L)
    N_lat, L = latents.shape

    # 2) 每个样本的“平均海温特征”
    train_ds = datamodule.train_data
    temp = train_ds.temperature  # 形状未知，大概率 (T, H, W)
    mask3d = train_ds.mask_3d  # 形状未知，大概率 (T, H, W) 或 (T, H, W, C)

    temp = temp.float()
    mask3d = mask3d.float()

    T_temp = temp.shape[0]
    T_mask = mask3d.shape[0]

    # 3) 三者取最小的时间长度，强制对齐
    T = min(N_lat, T_temp, T_mask)
    if T == 0:
        raise RuntimeError("用于计算 latent 统计的样本数为 0，请检查数据。")

    latents_used = latents[:T, :]  # (T, L)
    temp_used = temp[:T].contiguous()  # (T, ...)
    mask_used = mask3d[:T].contiguous()  # (T, ...)

    # 4) 展平空间维度，逐样本求掩膜内平均值
    temp_flat = temp_used.view(T, -1)  # (T, S)
    mask_flat = mask_used.view(T, -1)  # (T, S)

    # 避免 mask 全 0
    mask_sum = mask_flat.sum(dim=1).clamp_min(1.0)  # (T,)
    sum_val = (temp_flat * mask_flat).sum(dim=1)  # (T,)
    feat = (sum_val / mask_sum).cpu().numpy()  # (T,)

    # 5) 计算两个指标
    dis = compute_latent_dissimilarity(latents_used)
    corr_mean = compute_latent_data_corr(latents_used, feat)

    return dis, corr_mean


def save_results_and_plots(
        base_dir: str,
        latent_list: List[int],
        diss_runs: Dict[int, List[float]],
        nmse_runs: Dict[int, List[float]],
        corr_runs: Dict[int, List[float]]
):
    """
    把统计结果存成 npz，并画出三联图：
      (a) dissimilarity 柱状图
      (b) 多次 seed 的 NMSE 箱线图
      (c) latent–data 对应性柱状图
    说明：
      base_dir 就是 latent_sweep 这个总文件夹，
      三联图和 npz 直接保存在这里。
    """
    os.makedirs(base_dir, exist_ok=True)

    latent_arr = np.array(latent_list, dtype=int)
    diss_mean = np.array([np.mean(diss_runs[L]) for L in latent_list], dtype=float)
    corr_mean = np.array([np.mean(corr_runs[L]) for L in latent_list], dtype=float)
    nmse_mean = np.array([np.mean(nmse_runs[L]) for L in latent_list], dtype=float)

    # 保存数据
    np.savez(
        os.path.join(base_dir, "latent_sweep_stats.npz"),
        latent_sizes=latent_arr,
        dissimilarity=diss_mean,
        correspondence=corr_mean,
        nmse_mean=nmse_mean,
    )

    # 画图
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # (a) dissimilarity
    ax = axes[0]
    ax.bar(range(len(latent_list)), diss_mean)
    ax.set_xticks(range(len(latent_list)))
    ax.set_xticklabels(latent_list)
    ax.set_xlabel("latent dimension L")
    ax.set_ylabel("dissimilarity")
    ax.set_title("(a) Dissimilarity of latent codes")
    for i, v in enumerate(diss_mean):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    # (b) NMSE 箱型图
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

    # (c) correspondence
    ax = axes[2]
    ax.bar(range(len(latent_list)), corr_mean)
    ax.set_xticks(range(len(latent_list)))
    ax.set_xticklabels(latent_list)
    ax.set_xlabel("latent dimension L")
    ax.set_ylabel("mean |corr|")
    ax.set_title("(c) Correspondence between latents and mean field")
    for i, v in enumerate(corr_mean):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig_path = os.path.join(base_dir, "latent_sweep_summary.png")
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)


# ------------------------------------------------
# 主函数：循环 latent_size & seed 训练 + 分析
# ------------------------------------------------
@hydra.main(version_base="1.2", config_path="../../configs", config_name="default")
def main(cfg: DictConfig):
    # 只针对 MMGN / MMGN_NEW，提前检查一下
    model_name = cfg.datamodule.model_name
    if model_name not in ("MMGN", "MMGN_NEW"):
        raise ValueError(
            f"当前 datamodule.model_name = {model_name}，"
            f"latent_sweep.py 目前只支持 MMGN / MMGN_NEW。"
        )

    # 使用项目内固定的 latent 输出目录，与启动位置无关。
    base_run_dir = PROJECT_ROOT / "outputs" / "latent"
    # 所有 latent 扫描的结果统一放在这个总文件夹下
    sweep_dir = os.path.join(base_run_dir, "latent_sweep")
    os.makedirs(sweep_dir, exist_ok=True)

    # 用 dict 记录每个 L 下，多次 seed 的结果，方便后面做平均/boxplot
    diss_runs: Dict[int, List[float]] = defaultdict(list)
    nmse_runs: Dict[int, List[float]] = defaultdict(list)
    corr_runs: Dict[int, List[float]] = defaultdict(list)

    for L in LATENT_LIST:
        for seed in SEEDS:
            print(f"\n========== latent_size = {L}, seed = {seed} ==========")

            # 直接在 cfg 上改（这些 key 都已存在，不涉及结构变更）
            cfg.seed = seed
            cfg.model.params_model.latent_size = int(L)

            # 所有子结果目录都挂在 latent_sweep/ 下面
            save_dir_L = os.path.join(sweep_dir, f"latent_L{L}_seed{seed}")
            cfg.model.params_model.save_dir = save_dir_L
            os.makedirs(save_dir_L, exist_ok=True)

            # 设随机种子
            pl.seed_everything(seed)

            # ===== 实例化 DataModule =====
            datamodule: pl.LightningDataModule = instantiate(cfg.datamodule)
            datamodule.prepare_data()
            normalizer = datamodule.setup()

            # ===== 实例化 Lightning 模型（MMGNetModule / MMGN_NEW 对应的 Module）=====
            model: pl.LightningModule = instantiate(cfg.model, normalizer)

            # ===== callbacks & logger（和 main.py 一样的套路）=====
            callbacks: List[pl.Callback] = []
            if hasattr(cfg, "callback"):
                for _, cfg_cb in cfg.callback.items():
                    if "_target_" in cfg_cb:
                        callbacks.append(instantiate(cfg_cb))

            logger = None
            if hasattr(cfg, "logger"):
                for _, cfg_logger in cfg.logger.items():
                    if "_target_" in cfg_logger:
                        logger = instantiate(cfg_logger)

            trainer = pl.Trainer(**cfg.trainer, callbacks=callbacks, logger=logger)

            # ===== 训练 =====
            trainer.fit(model=model, datamodule=datamodule)

            # ===== 评估：NMSE、latent 统计 =====
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)

            nmse = evaluate_nmse_on_test(model, datamodule)
            dis, corr_mean = compute_latent_stats(model, datamodule)

            print(f"[L={L}, seed={seed}] NMSE = {nmse:.6e}, "
                  f"dissimilarity = {dis:.4f}, corr_mean = {corr_mean:.4f}")

            nmse_runs[L].append(nmse)
            diss_runs[L].append(dis)
            corr_runs[L].append(corr_mean)

            if logger is not None and hasattr(logger, "experiment"):
                try:
                    logger.experiment.finish()
                except Exception:
                    pass

            # 释放资源
            del datamodule, normalizer, model, trainer, callbacks, logger
            gc.collect()
            torch.cuda.empty_cache()

    # ===== 所有 (L, seed) 跑完，保存结果 + 画图 =====
    save_results_and_plots(
        base_dir=sweep_dir,
        latent_list=LATENT_LIST,
        diss_runs=diss_runs,
        nmse_runs=nmse_runs,
        corr_runs=corr_runs,
    )
    print(f"\n全部完成，结果和图已保存在：{sweep_dir}")


if __name__ == "__main__":
    main()
