# eval_ckpt.py
# -*- coding: utf-8 -*-
"""
功能：
- 输入：已训练好的 .ckpt 文件路径
- 输出：
    1) 模型参数量（总参数量 & 可训练参数量）
    2) 单次推理时间（单个 batch 的平均推理时间 + 单样本平均时间）
    3) 在测试集上的 MSE

注意：
- 默认使用 current.yaml 作为 DataModule 配置，你可以通过 --data-cfg 指定其他 YAML。
- 请在项目根目录下运行本脚本。
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import time

import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate

# 先导入 DataModule，间接把 dataloaders.* 等都 import 进来，方便反序列化 normalizer 等对象
from data_interface import DataModule  # noqa: F401
from model_interface import BaselineModule, MMGNetModule


def parse_args():
    parser = argparse.ArgumentParser(
        description="评估已训练 ckpt 的参数量、推理时间和测试集 MSE"
    )

    # ====== 这里用“带名字”的可选参数，并给出你的默认路径 ======
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="已训练好的 Lightning ckpt 文件路径",
    )

    parser.add_argument(
        "--data-cfg",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "datamodule" / "current.yaml"),
        help="DataModule 的配置 YAML（包含 _target_: data_interface.DataModule）",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="'auto' / 'cpu' / 'cuda' / 'cuda:0' 等",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="推理时间测试的预热次数",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=50,
        help="推理时间测试的重复次数，用于取平均",
    )
    return parser.parse_args()


def resolve_device(device_flag: str) -> str:
    if device_flag == "auto":
        if torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    return device_flag


def load_model_from_ckpt(ckpt_path: str, device: str) -> torch.nn.Module:
    """
    根据 ckpt 中 hyper_parameters 自动判断是 BaselineModule 还是 MMGNetModule，
    并恢复模型参数。
    """
    # 为了判断类型，先在 CPU 上加载一次
    ckpt = torch.load(ckpt_path, map_location="cpu")
    hparams = ckpt.get("hyper_parameters", {})

    # MMGNetModule 在 hparams 里会多一个 params_data
    if "params_data" in hparams:
        cls = MMGNetModule
    else:
        cls = BaselineModule

    # 使用 Lightning 提供的 load_from_checkpoint 恢复模型
    model = cls.load_from_checkpoint(ckpt_path, map_location=device)
    model.to(device)
    model.eval()
    return model


def build_datamodule(cfg_path: str) -> DataModule:
    """
    使用 OmegaConf + hydra.instantiate 构建 DataModule。
    YAML 中需要包含：
        _target_: data_interface.DataModule
    """
    cfg = OmegaConf.load(cfg_path)
    datamodule: DataModule = instantiate(cfg)
    datamodule.prepare_data()
    datamodule.setup("test")
    return datamodule


def count_parameters(model: torch.nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def move_batch_to_device(batch, device: str):
    """
    将 batch 中的张量移动到指定 device，其它对象保持不变。
    """
    batch_on_device = []
    for item in batch:
        if torch.is_tensor(item):
            batch_on_device.append(item.to(device, non_blocking=True))
        else:
            batch_on_device.append(item)
    return batch_on_device


def measure_inference_time(
    model,
    batch,
    device: str,
    warmup: int = 10,
    repeat: int = 50,
):
    """
    使用 model.step(batch) 来估计推理时间：
    - 预热 warmup 次
    - 重复 repeat 次计算平均单次时间
    """
    is_cuda = device.startswith("cuda") and torch.cuda.is_available()

    batch_on_device = move_batch_to_device(batch, device)

    model.eval()
    with torch.no_grad():
        # 预热
        for _ in range(warmup):
            _ = model.step(batch_on_device)
        if is_cuda:
            torch.cuda.synchronize()

        # 正式计时
        start = time.perf_counter()
        for _ in range(repeat):
            _ = model.step(batch_on_device)
        if is_cuda:
            torch.cuda.synchronize()
        end = time.perf_counter()

    avg_time = (end - start) / float(repeat)

    # 根据 y 的形状估计 batch_size
    y = batch_on_device[1]
    # y: (B, N, 1)
    if y.ndim >= 1:
        batch_size = y.shape[0]
    else:
        batch_size = 1

    avg_per_sample = avg_time / float(batch_size)
    return avg_time, avg_per_sample


def evaluate_test_mse(model, datamodule, device: str) -> float:
    """
    手工在 test_dataloader 上循环：
    使用 model.step(batch) 得到 batch 的 MSE，
    按元素个数加权平均，得到全测试集的真实 MSE。
    """
    model.eval()
    loader = datamodule.test_dataloader()

    total_mse_times_n = 0.0
    total_n = 0

    with torch.no_grad():
        for batch in loader:
            batch_on_device = move_batch_to_device(batch, device)
            loss, batch_mse, yhat, yref, mask = model.step(batch_on_device)

            # yref: (B, N, 1)
            num_elements = yref.numel()
            total_mse_times_n += batch_mse.item() * num_elements
            total_n += num_elements

    if total_n == 0:
        return float("nan")
    return total_mse_times_n / float(total_n)


def main():
    args = parse_args()
    device = resolve_device(args.device)

    print(f"== 设备选择: {device} ==")
    print(f"加载 ckpt: {args.ckpt}")
    print(f"DataModule 配置: {args.data_cfg}")

    # 1. 构建 DataModule（用于测试集）
    datamodule = build_datamodule(args.data_cfg)

    # 2. 从 ckpt 恢复模型
    model = load_model_from_ckpt(args.ckpt, device)

    # 3. 统计参数量
    total_params, trainable_params = count_parameters(model)
    print("\n=== 模型参数量 ===")
    print(f"  总参数量        : {total_params:,d}")
    print(f"  可训练参数量    : {trainable_params:,d}")
    print(f"  总参数量 (M)    : {total_params / 1e6:.3f} M")
    print(f"  可训练参数量 (M): {trainable_params / 1e6:.3f} M")

    # 4. 推理时间（单 batch）
    test_loader = datamodule.test_dataloader()
    try:
        first_batch = next(iter(test_loader))
    except StopIteration:
        raise RuntimeError("测试集为空，无法测量推理时间和计算 MSE。")

    avg_step_time, avg_sample_time = measure_inference_time(
        model,
        first_batch,
        device,
        warmup=args.warmup,
        repeat=args.repeat,
    )

    print("\n=== 推理耗时估计 ===")
    print(f"  单 batch 平均推理时间: {avg_step_time * 1000:.3f} ms")
    print(f"  单样本平均推理时间  : {avg_sample_time * 1000:.3f} ms")

    # 5. 全测试集 MSE
    test_mse = evaluate_test_mse(model, datamodule, device)
    print("\n=== 测试集 MSE ===")
    print(f"  Test MSE: {test_mse:.6e}")


if __name__ == "__main__":
    main()
