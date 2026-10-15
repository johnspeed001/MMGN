# -*- coding: utf-8 -*-
"""
save_npz_meta_to_txt.py
---------------------------------------------
读取 npz 文件并自动将元信息 (meta + shape)
保存为同路径下的 .txt 文件。

✅ 直接右键运行即可，无需命令行参数。
---------------------------------------------
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import numpy as np
import os

def save_npz_meta(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ 找不到文件: {file_path}")

    # 载入 npz 数据
    data = np.load(file_path, allow_pickle=True)
    print(f"\n✅ 已载入文件: {file_path}")

    # 提取字段
    lats = data["lats"]
    lons = data["lons"]
    temperature = data["temperature"]
    meta = data["meta"][0] if "meta" in data else {}

    # 生成统计信息
    info_lines = []
    info_lines.append(f"文件路径: {file_path}")
    info_lines.append(f"数据维度: temperature = {temperature.shape}")
    info_lines.append(f"纬度范围: {lats.min():.2f} ~ {lats.max():.2f}  (共 {len(lats)} 点)")
    info_lines.append(f"经度范围: {lons.min():.2f} ~ {lons.max():.2f}  (共 {len(lons)} 点)")
    info_lines.append(f"值范围: {temperature.min():.4f} ~ {temperature.max():.4f}\n")

    if meta:
        info_lines.append("【meta 信息】")
        for k, v in meta.items():
            info_lines.append(f"{k}: {v}")

    # 保存路径
    save_path = os.path.splitext(file_path)[0] + "_meta.txt"

    # 写入文件
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(info_lines))

    print(f"📝 已保存元信息到: {save_path}")


if __name__ == "__main__":
    # 🟩 默认 npz 文件路径（可自行修改）
    file_path = str(PROJECT_ROOT / "data" / "current" / "bohai.npz")
    save_npz_meta(file_path)
