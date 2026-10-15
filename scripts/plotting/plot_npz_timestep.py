# -*- coding: utf-8 -*-
"""
plot_npz_timestep.py
--------------------------------------------------------
根据输入的时间步 (--step) 绘制 npz 文件中对应帧的数据图。
适用于 CMEMS / SST / GST / Ocean Twin 数据可视化。

✅ 支持命令行运行：
    python plot_npz_timestep.py --input data/test/cmems_current_velocity.npz --step 10

✅ 也可直接右键运行（默认 step=0）
--------------------------------------------------------
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def plot_timestep(npz_path, step, output=None):
    # 检查文件
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"❌ 文件不存在: {npz_path}")

    # 载入数据
    data = np.load(npz_path, allow_pickle=True)
    lats = data["lats"]
    lons = data["lons"]
    field = data["temperature"]
    meta = data["meta"][0] if "meta" in data else None

    T, H, W = field.shape
    if step < 0 or step >= T:
        raise ValueError(f"❌ step 超出范围 (0 ~ {T-1})")

    print(f"✅ 文件: {npz_path}")
    print(f"→ 时间步: {step}/{T-1}")
    print(f"→ 数据维度: {field.shape}")

    # 当前帧
    frame = field[step, :, :]

    # 绘制
    plt.figure(figsize=(8, 6))
    plt.imshow(frame, cmap="turbo", origin="lower",
               extent=[lons.min(), lons.max(), lats.min(), lats.max()])
    plt.colorbar(label="Value")
    plt.xlabel("Longitude (°E)")
    plt.ylabel("Latitude (°N)")

    title = f"Frame {step}/{T-1}"
    if meta and "variable" in meta:
        title += f" — {meta['variable']}"
    plt.title(title)

    # 自动保存输出
    save_path = Path(output) if output else PROJECT_ROOT / "outputs" / "figures" / f"{Path(npz_path).stem}_step{step}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"🖼️ 已保存图像: {save_path}")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot a timestep from an .npz dataset")
    parser.add_argument("--input", type=str, default=str(PROJECT_ROOT / "data" / "current" / "bohai.npz"),
                        help="输入 npz 文件路径")
    parser.add_argument("--step", type=int, default=10,
                        help="时间步索引 (0-based)")
    parser.add_argument("--output", help="Output PNG path (default: outputs/figures)")
    args = parser.parse_args()

    plot_timestep(args.input, args.step, args.output)
