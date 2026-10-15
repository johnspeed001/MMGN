# -*- coding: utf-8 -*-
"""
convert_cmems_current_to_npz_uv.py
---------------------------------------------
将 Copernicus CMEMS 海洋流场数据 (uo, vo) 分别转换为两个 npz 文件：
  - u 文件: temperature=uo (按 fill_mode 处理)
  - v 文件: temperature=vo (按 fill_mode 处理)
  - 同一份有效性掩膜 mask：海=1，陆/无数据=0（定义为 uo 和 vo 同时有效）
  - 字段名保持不变: lats, lons, temperature, mask, meta

fill_mode:
  - "zero": 无效点填 0（推荐，稳定）
  - "nan" : 保留 NaN（需下游支持）
  - "mean": 用每个时间片的海域均值填充（分别对 u、v 处理）
"""

import numpy as np
import xarray as xr
import argparse
import os
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def _fill_by_mode(arr, valid_mask, mode: str):
    """对形状 (T,H,W) 的分量数组按 fill_mode 进行填充。"""
    if mode == "zero":
        return np.where(valid_mask, arr, 0.0)
    if mode == "nan":
        return np.where(valid_mask, arr, np.nan)
    if mode == "mean":
        filled = arr.copy()
        T, H, W = filled.shape
        for t in range(T):
            m = valid_mask[t]
            if m.any():
                mean_val = np.nanmean(arr[t][m])
                filled[t][~m] = mean_val
            else:
                filled[t][~m] = 0.0
        return filled
    raise ValueError(f"Unknown fill_mode: {mode}")

def convert_cmems_to_uv(input_path, out_u_path, out_v_path, fill_mode="zero"):
    ds = xr.open_dataset(input_path)

    # 基本变量
    lats = ds["latitude"].values          # (H,)
    lons = ds["longitude"].values         # (W,)
    uo = ds["uo"].values                  # (T, D, H, W)
    vo = ds["vo"].values                  # (T, D, H, W)
    time = ds["time"].values              # (T,)

    # 只取最上层
    uo = uo[:, 0, :, :]                   # (T,H,W)
    vo = vo[:, 0, :, :]                   # (T,H,W)

    # 统一掩膜：u 与 v 同时为有限值的点作为海域
    valid_mask = np.isfinite(uo) & np.isfinite(vo)     # (T,H,W)

    # 分别填充 u、v
    filled_u = _fill_by_mode(uo, valid_mask, fill_mode)
    filled_v = _fill_by_mode(vo, valid_mask, fill_mode)

    # 元数据（统一）
    T, H, W = filled_u.shape
    dlat = float(abs(lats[1] - lats[0])) if len(lats) > 1 else np.nan
    dlon = float(abs(lons[1] - lons[0])) if len(lons) > 1 else np.nan
    ocean_ratio = float(valid_mask.sum() / valid_mask.size)

    def make_meta(var_name: str):
        return np.array([{
            "time_steps": int(T),
            "lat_points": int(H),
            "lon_points": int(W),
            "resolution_lat": dlat,
            "resolution_lon": dlon,
            "time_info": f"{str(time[0])} → {str(time[-1])}",
            "variable": var_name,                    # 仅在 meta 里标识具体分量
            "fill_mode": fill_mode,
            "ocean_ratio": ocean_ratio,
            "source_file": os.path.basename(input_path),
            "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }], dtype=object)

    # 写出 u 文件
    os.makedirs(os.path.dirname(out_u_path), exist_ok=True)
    np.savez_compressed(
        out_u_path,
        lats=lats.astype(np.float32),
        lons=lons.astype(np.float32),
        temperature=filled_u.astype(np.float32),   # 仍然叫 temperature，但存放的是 u
        mask=valid_mask.astype(np.uint8),
        meta=make_meta("uo (zonal current, stored in 'temperature')"),
    )

    # 写出 v 文件
    os.makedirs(os.path.dirname(out_v_path), exist_ok=True)
    np.savez_compressed(
        out_v_path,
        lats=lats.astype(np.float32),
        lons=lons.astype(np.float32),
        temperature=filled_v.astype(np.float32),   # 仍然叫 temperature，但存放的是 v
        mask=valid_mask.astype(np.uint8),
        meta=make_meta("vo (meridional current, stored in 'temperature')"),
    )

    # 打印信息
    print(f"\n✅ 转换完成：")
    print(f"U 文件 → {out_u_path}")
    print(f"V 文件 → {out_v_path}")
    print(f"→ 数据维度: (time={T}, lat={H}, lon={W})")
    print(f"→ 纬度范围: {lats.min():.2f} ~ {lats.max():.2f}")
    print(f"→ 经度范围: {lons.min():.2f} ~ {lons.max():.2f}")
    print(f"→ 时间范围: {str(time[0])} → {str(time[-1])}")
    print(f"→ 空间分辨率: Δlat={dlat:.3f}, Δlon={dlon:.3f}")
    print(f"→ 海域占比(ocean_ratio): {ocean_ratio:.4f}")
    print(f"→ 填充策略(fill_mode): {fill_mode}")

    ds.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str,
                        required=True,
                        help="CMEMS NetCDF 文件路径")
    parser.add_argument("--output", type=str,
                        default=str(PROJECT_ROOT / "data" / "processed" / "current" / "velocity.npz"),
                        help="输出基名（不含 _u/_v 后缀），程序会写出 *_u.npz 与 *_v.npz")
    parser.add_argument("--fill_mode", type=str, default="zero",
                        choices=["zero", "nan", "mean"],
                        help="缺失值填充策略")
    args = parser.parse_args()

    base = Path(args.output)
    out_u = str(base.with_name(base.stem + "_u").with_suffix(".npz"))
    out_v = str(base.with_name(base.stem + "_v").with_suffix(".npz"))

    convert_cmems_to_uv(args.input, out_u, out_v, fill_mode=args.fill_mode)
