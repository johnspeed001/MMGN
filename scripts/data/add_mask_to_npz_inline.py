# -*- coding: utf-8 -*-
"""
add_mask_to_npz_inline.py
-------------------------
单文件、可右键/双击运行：直接在代码里指定 .npz 文件路径，为其中的“主数据数组”生成掩码并写回为新文件。
- 不修改原始 .npz（默认在同目录生成 "<原名>.with_mask.npz"）。
- 掩码 key 固定为 "mask"，取值 1=有效、0=无效（uint8），与项目已有脚本 convert_nc_to_npz.py 保持一致。
- 支持 2D/3D 数组（(H,W) 或 (T,H,W)）。

使用：
  1) 修改下方的 INPUT_PATH（必填）。
  2) 如需指定哪个键生成掩码，设置 DATA_KEY（可选；默认自动挑选 2D/3D 最大的那个数组）。
  3) 双击或右键用 Python 运行即可。

如需“原地覆盖”：将 REPLACE_INPLACE=True（会先把原文件重命名为 .bak，再生成同名新文件）。
"""

import numpy as np
from pathlib import Path
import sys
from typing import Optional, Dict, Tuple

# ============== 用户需要改的区域（至少改 INPUT_PATH） ==============
INPUT_PATH      = str(Path(__file__).resolve().parents[2] / "data" / "current" / "bohai.npz")  # 必填：你的 .npz 文件路径，例如 r"D:\data\sst_region.npz"
DATA_KEY: Optional[str] = "temperature"  # 可选：主数据数组的键名（例如 "temperature" / "data"）。留空则自动猜
REPLACE_INPLACE = False         # True=原地覆盖（原文件会先改名为 .bak）；False=写 <stem>.with_mask.npz
FILL_VALUE: Optional[float] = None  # 可选：把等于该值的像素视为无效（例如 0 或 -9999）；默认 None 不额外剔除
# ===============================================================

def _is_2d_or_3d_numeric(a: np.ndarray) -> bool:
    return isinstance(a, np.ndarray) and a.ndim in (2, 3) and np.issubdtype(a.dtype, np.number)

def _choose_data_key(npz: Dict[str, np.ndarray], prefer: Optional[str]) -> str:
    if prefer and prefer in npz:
        if _is_2d_or_3d_numeric(npz[prefer]):
            return prefer
        else:
            raise ValueError(f"指定 DATA_KEY='{prefer}' 不是 2D/3D 数组（或非数值），shape={npz[prefer].shape}, dtype={npz[prefer].dtype}")
    # 自动挑选：所有 2D/3D 数组里，选“元素个数最多”的一个
    candidates = [(k, v) for k, v in npz.items() if _is_2d_or_3d_numeric(v)]
    if not candidates:
        raise ValueError("该 NPZ 中没有可用的 2D/3D 数组键。请设置 DATA_KEY。")
    candidates.sort(key=lambda kv: kv[1].size, reverse=True)
    return candidates[0][0]

def _infer_mask(arr: np.ndarray, fill_value: Optional[float]) -> np.ndarray:
    # 1=有效（finite and != fill_value），0=无效；dtype=uint8；形状与 arr 相同
    m = np.isfinite(arr)
    if fill_value is not None:
        try:
            m &= (arr != fill_value)
        except Exception:
            pass
    mask = m.astype(np.uint8, copy=False)
    return np.ascontiguousarray(mask)

def _write_npz(out_path: Path, arrays: Dict[str, np.ndarray]):
    # 保持压缩；字段顺序对训练无影响
    np.savez_compressed(out_path, **arrays)

def main():
    in_path = Path(INPUT_PATH)
    if not in_path.exists() or in_path.suffix.lower() != ".npz":
        raise FileNotFoundError("请在脚本顶部填写正确的 .npz 路径（INPUT_PATH）。")

    loaded = np.load(in_path, allow_pickle=True)  # 兼容含 meta=object 的情况
    arrays = {k: loaded[k] for k in loaded.files}  # 完整复制，不动原字段
    loaded.close()

    key = _choose_data_key(arrays, DATA_KEY)
    arr = arrays[key]
    if arr.ndim not in (2, 3):
        raise ValueError(f"DATA_KEY='{key}' 的数组维度不支持：shape={arr.shape}")

    mask = _infer_mask(arr, FILL_VALUE)  # uint8，形状与 arr 相同
    arrays["mask"] = mask  # 与项目约定保持一致的字段名

    if REPLACE_INPLACE:
        bak_path = in_path.with_suffix(".npz.bak")
        in_path.rename(bak_path)
        out_path = in_path  # 与原名一致
        _write_npz(out_path, arrays)
        print(f"已原位写入：{out_path}（原文件已备份为：{bak_path}）")
    else:
        out_path = in_path.with_name(in_path.stem + ".with_mask.npz")
        _write_npz(out_path, arrays)
        print(f"已生成：{out_path}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("出错：", e)
        raise
