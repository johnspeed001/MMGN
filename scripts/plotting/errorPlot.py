from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import os
import numpy as np
import matplotlib.pyplot as plt
# ================== 参数区 ==================
DATA_NPZ = str(PROJECT_ROOT / "data" / "current" / "bohai.npz")
PRED_PATH = None  # Set by --run-dir.
TGT_PATH = None

SAMPLE_IDX = 3  # 第几个样本
SAVE_FIG = str(PROJECT_ROOT / "outputs" / "figures" / "error_map.png")

ERR_VMAX = 0.40  # 色条统一上限（比如 0.40）
# 如果想自动根据数据设置色条上限，就把 ERR_VMAX 设为 None，改用分位数
ERR_PCT = 99  # ERR_VMAX 为 None 时使用
# ========================================
def load_mask(npz_path, T_need, H, W):
    dat = np.load(npz_path)
    if "mask" in dat.files:
        m = dat["mask"]
        if m.ndim == 2:
            m = np.broadcast_to(m[None, :, :], (T_need, H, W))
        elif m.ndim == 3:
            if m.shape[0] < T_need:
                raise ValueError(f"mask 时间长度 {m.shape[0]} < 需要的 {T_need}")
            m = m[:T_need]
        else:
            raise ValueError("mask 必须是 2D 或 3D")
        return m.astype(bool)
    else:
        return np.ones((T_need, H, W), dtype=bool)


def reshape_to_THW(arr, H, W):
    """统一成 (T, H, W)"""
    arr = np.asarray(arr)

    if arr.ndim == 4:  # (nB, B, N, 1)
        n1, n2, N, C = arr.shape
        if C != 1:
            raise ValueError(f"最后一维应为 1，得到 {C}")
        T = n1 * n2
        if N != H * W:
            raise ValueError(f"N={N} 与 H*W={H * W} 不一致")
        arr = arr.reshape(T, N)

    if arr.ndim == 3:
        T, N, C = arr.shape
        if C == 1:  # (T, N, 1)
            arr = arr.reshape(T, N)
        else:
            if (N, C) == (H, W):  # (T, H, W)
                return arr
            raise ValueError(f"无法解析的 3D 形状: {arr.shape}")

    if arr.ndim == 2:  # (T, H*W)
        T, N = arr.shape
        if N != H * W:
            raise ValueError(f"N={N} 与 H*W={H * W} 不一致")
        return arr.reshape(T, H, W)

    raise ValueError(f"暂不支持的数组形状: {arr.shape}")


def main():
    # 网格尺寸
    dat0 = np.load(DATA_NPZ)
    lats = dat0["lats"]
    lons = dat0["lons"]
    H, W = len(lats), len(lons)
    print(f"[INFO] 网格尺寸: H={H}, W={W}")

    # 预测 / 真值
    pred_raw = np.load(PRED_PATH)
    tgt_raw = np.load(TGT_PATH)
    tgt_all = reshape_to_THW(tgt_raw, H, W)
    pred_all = reshape_to_THW(pred_raw, H, W)

    if tgt_all.shape != pred_all.shape:
        raise ValueError(f"pred 和 tgt 形状不一致: {pred_all.shape} vs {tgt_all.shape}")

    T = tgt_all.shape[0]
    if SAMPLE_IDX < 0 or SAMPLE_IDX >= T:
        raise ValueError(f"SAMPLE_IDX={SAMPLE_IDX} 超出范围 [0,{T - 1}]")

    # mask
    mask_3d = load_mask(DATA_NPZ, T, H, W)
    mask = mask_3d[SAMPLE_IDX]  # (H,W) bool

    # 该时刻场 + 掩膜
    tgt = tgt_all[SAMPLE_IDX]
    pred = pred_all[SAMPLE_IDX]
    tgt_masked = np.where(mask, tgt, np.nan)
    pred_masked = np.where(mask, pred, np.nan)

    # 绝对误差（归一化空间）
    err = np.abs(pred_masked - tgt_masked)

    # 统一色条上限
    if ERR_VMAX is None:
        vmax_err = np.nanpercentile(err, ERR_PCT)
    else:
        vmax_err = ERR_VMAX

    # 配色：viridis + 陆地灰色
    cmap_err = plt.get_cmap("viridis").copy()
    cmap_err.set_bad(color="lightgray")

    fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
    im = ax.imshow(err, origin="lower", cmap=cmap_err,
                   vmin=0.0, vmax=vmax_err)
    ax.set_xticks([])
    ax.set_yticks([])

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Absolute error (normalized)", fontsize=8)
    # 可选：固定几个刻度
    cbar.set_ticks([0.0, 0.1, 0.2, 0.3, 0.4])

    plt.tight_layout()
    Path(SAVE_FIG).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(SAVE_FIG, bbox_inches="tight")
    print(f"[INFO] 误差图已保存到: {os.path.abspath(SAVE_FIG)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot prediction error for a training run")
    parser.add_argument("--run-dir", type=Path, required=True, help="Directory containing predictions.npy and targets.npy")
    parser.add_argument("--data", default=DATA_NPZ)
    parser.add_argument("--output", default=SAVE_FIG)
    parser.add_argument("--sample", type=int, default=SAMPLE_IDX)
    args = parser.parse_args()
    DATA_NPZ, SAVE_FIG, SAMPLE_IDX = args.data, args.output, args.sample
    PRED_PATH = args.run_dir / "predictions.npy"
    TGT_PATH = args.run_dir / "targets.npy"
    main()
