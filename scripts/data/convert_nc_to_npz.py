# -*- coding: utf-8 -*-
"""
convert_cmems_current_to_npz_masked.py
---------------------------------------------
将 Copernicus CMEMS 海洋流场数据 (uo, vo) 转换为 npz：
  - 保存速度模值 (filled)
  - 同时保存有效性掩膜 mask：海=1，陆/无数据=0
  - 兼容后续“只在 mask==1 的点上计算 loss/metric”的训练流程
输出字段:
    lats, lons, temperature(=√(uo²+vo²) after fill), mask, meta
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import numpy as np
import xarray as xr
import argparse
import os
from datetime import datetime

def convert_cmems(input_path, output_path, fill_mode="zero"):
    """
    fill_mode:
        - "zero":  将无效点填 0（推荐，稳定）
        - "nan":   保留 NaN（需下游代码支持 mask + 忽略 NaN）
        - "mean":  用每个时间片的海域均值填充
    """
    ds = xr.open_dataset(input_path)

    # 基本变量
    lats = ds["latitude"].values          # (H,)
    lons = ds["longitude"].values         # (W,)
    uo = ds["uo"].values                  # (T, D, H, W)
    vo = ds["vo"].values                  # (T, D, H, W)
    time = ds["time"].values              # (T,)

    # 最上层
    uo = uo[:, 0, :, :]                   # (T,H,W)
    vo = vo[:, 0, :, :]                   # (T,H,W)

    # 有效性掩膜：uo/vo 都是有限值 -> 1
    valid_mask = np.isfinite(uo) & np.isfinite(vo)     # (T,H,W)

    # 速度模值（先计算，保留 NaN）
    speed = np.sqrt(uo**2 + vo**2)                     # (T,H,W)

    # 依据 fill_mode 生成“可训练”的 filled_speed
    if fill_mode == "zero":
        filled_speed = np.where(valid_mask, speed, 0.0)
    elif fill_mode == "nan":
        filled_speed = np.where(valid_mask, speed, np.nan)
    elif fill_mode == "mean":
        filled_speed = speed.copy()
        T, H, W = filled_speed.shape
        for t in range(T):
            m = valid_mask[t]
            if m.any():
                mean_val = np.nanmean(speed[t][m])
                filled_speed[t][~m] = mean_val
            else:
                filled_speed[t][~m] = 0.0
    else:
        raise ValueError(f"Unknown fill_mode: {fill_mode}")

    # 元数据
    T, H, W = filled_speed.shape
    dlat = float(abs(lats[1] - lats[0])) if len(lats) > 1 else np.nan
    dlon = float(abs(lons[1] - lons[0])) if len(lons) > 1 else np.nan
    ocean_ratio = float(valid_mask.sum() / valid_mask.size)

    meta = {
        "time_steps": int(T),
        "lat_points": int(H),
        "lon_points": int(W),
        "resolution_lat": dlat,
        "resolution_lon": dlon,
        "time_info": f"{str(time[0])} → {str(time[-1])}",
        "variable": "velocity_magnitude (sqrt(uo^2 + vo^2))",
        "fill_mode": fill_mode,
        "ocean_ratio": ocean_ratio,
        "source_file": os.path.basename(input_path),
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # 输出
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(
        output_path,
        lats=lats.astype(np.float32),
        lons=lons.astype(np.float32),
        temperature=filled_speed.astype(np.float32),  # 喂给网络
        mask=valid_mask.astype(np.uint8),            # 训练/评估用掩膜
        meta=np.array([meta], dtype=object),
    )

    print(f"\n✅ 转换完成: {output_path}")
    print(f"→ 数据维度: (time={T}, lat={H}, lon={W})")
    print(f"→ 纬度范围: {lats.min():.2f} ~ {lats.max():.2f}")
    print(f"→ 经度范围: {lons.min():.2f} ~ {lons.max():.2f}")
    print(f"→ 时间范围: {meta['time_info']}")
    print(f"→ 空间分辨率: Δlat={dlat:.3f}, Δlon={dlon:.3f}")
    print(f"→ 海域占比(ocean_ratio): {ocean_ratio:.4f}")
    print(f"→ 填充策略(fill_mode): {fill_mode}")
    ds.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CMEMS NetCDF to masked velocity NPZ")
    parser.add_argument("--input", required=True, help="NetCDF path, normally under data/raw/cmems")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "data" / "processed" / "current" / "velocity.npz"))
    parser.add_argument("--fill_mode", default="zero", choices=["zero", "nan", "mean"])
    args = parser.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    convert_cmems(args.input, args.output, fill_mode=args.fill_mode)
