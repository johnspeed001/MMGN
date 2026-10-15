# -*- coding: utf-8 -*-
"""
Latent Dimension Ablation (Final Stable Version)
Fully supports:
- Lightning CKPT
- archive/data.pkl with persistent storages
- PyTorch old-storage API (your version)
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import matplotlib.pyplot as plt
import zipfile, pickle, io

from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

###########################################################################
# CKPT List
###########################################################################
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

###########################################################################
# Import Model & DataModule
###########################################################################
from data_interface import DataModule
from models.MMGNet_net import MMGNet

###########################################################################
# Metrics
###########################################################################
_EPS = 1e-12
ssim_metric = StructuralSimilarityIndexMeasure().to(DEVICE)

def masked_mse(pred, gt, mask):
    return ((pred - gt) ** 2 * mask).sum() / mask.sum().clamp_min(_EPS)

def compute_psnr(pred, gt, mask):
    mse = masked_mse(pred, gt, mask)
    valid = gt[mask.bool()]
    dr = (valid.max() - valid.min()).clamp_min(_EPS)
    return (10 * torch.log10((dr*dr) / mse)).item()

def compute_ssim(pred, gt, mask, H, W):
    pred = pred.reshape(1,1,H,W)
    gt   = gt.reshape(1,1,H,W)
    mask = mask.reshape(1,1,H,W)
    pred = gt * (1-mask) + pred * mask
    return ssim_metric(pred, gt).item()

###########################################################################
# Persistent Loader (Final stable version for your PyTorch)
###########################################################################
class PersistentLoadUnpickler(pickle.Unpickler):
    """
    Supports all Lightning CKPT persistent storage formats:
    - ('storage', type, key, size)
    - ('storage', type, key, device, size)
    - ('storage', type, key, device, size, view)
    Works on old PyTorch + Windows.
    """

    def __init__(self, file, zip_file):
        super().__init__(file)
        self.zip_file = zip_file
        self.storage_cache = {}

    def persistent_load(self, pid):

        if pid[0] != "storage":
            raise pickle.UnpicklingError(f"Unsupported pid: {pid}")

        # unpack possible formats
        if len(pid) == 4:
            storage_type, key, size = pid[1:]
        elif len(pid) == 5:
            storage_type, key, _, size = pid[1:]
        elif len(pid) == 6:
            storage_type, key, _, size, _ = pid[1:]
        else:
            raise pickle.UnpicklingError(f"Unknown pid format: {pid}")

        # read raw bytes (Lightning stored tensor storage here)
        raw_bytes = self.zip_file.read(f"archive/data/{key}")
        byte_arr = bytearray(raw_bytes)

        # element size
        tmp_tensor = torch.tensor([], dtype=storage_type.dtype)
        elem_size = tmp_tensor.element_size()
        total_bytes = size * elem_size

        # Allocate UntypedStorage
        storage = torch.UntypedStorage(total_bytes)

        # Convert raw bytes -> ByteTensor
        byte_tensor = torch.tensor(list(byte_arr), dtype=torch.uint8)

        # Copy byte_tensor's storage into storage
        # (older PyTorch supports copy_ on storages)
        storage.copy_(byte_tensor.storage())

        return storage


def safe_load_state_dict(ckpt_path):

    with zipfile.ZipFile(ckpt_path, "r") as z:
        raw_pkl = z.read("archive/data.pkl")
        up = PersistentLoadUnpickler(io.BytesIO(raw_pkl), z)
        data = up.load()

    sd = data["state_dict"]
    clean = {k.replace("model.", ""): v for k, v in sd.items()}
    return clean

###########################################################################
# Build Model (按照你的 YAML)
###########################################################################
def build_model(latent_size):
    return MMGNet(
        in_size=2,
        hidden_size=256,
        n_data=61,
        latent_size=latent_size,
        latent_init="zeros",
        out_size=1,
        n_layers=5,
        input_scale=256,
        alpha=1,
        filter="Gabor",
    ).to(DEVICE)


###########################################################################
# Latent Ablation
###########################################################################
@torch.no_grad()
def ablate_latent_dims(model, coords, target, mask, idx, H, W):

    latent = model.latents[idx].detach().clone()
    L = latent.shape[0]

    pred0 = model(coords, torch.tensor([idx], device=DEVICE))
    mse0  = masked_mse(pred0, target, mask).item()
    psnr0 = compute_psnr(pred0, target, mask)
    ssim0 = compute_ssim(pred0, target, mask, H, W)

    mse_list, psnr_list, ssim_list = [], [], []

    for j in range(L):
        z = latent.clone()
        z[j] = 0
        model.latents.data[idx] = z

        pred = model(coords, torch.tensor([idx], device=DEVICE))
        mse_list.append(masked_mse(pred, target, mask).item())
        psnr_list.append(compute_psnr(pred, target, mask))
        ssim_list.append(compute_ssim(pred, target, mask, H, W))

    model.latents.data[idx] = latent

    return mse0, mse_list, psnr0, psnr_list, ssim0, ssim_list


###########################################################################
# MAIN
###########################################################################
def main():
    out_dir = PROJECT_ROOT / "outputs" / "latent" / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    dm = DataModule(
        data_dir=str(PROJECT_ROOT / "data" / "current" / "bohai.npz"),
        model_name="MMGN",
        pre_method=["minmax","minmax"],
        reduce_dim=[[],[0]],
        task="task1",
        sampling_rate=[0.5,0.5],
        n_train_val=[61,6],
        b_train_val=[1,1],
    )
    dm.setup()

    coords, target, idx, mask = next(iter(dm.test_dataloader()))
    coords, target, mask = coords.to(DEVICE), target.to(DEVICE), mask.to(DEVICE)
    idx = int(idx.item())
    _, H, W = dm.test_data.temperature.shape

    cursor = 0
    mse_all, psnr_all, ssim_all = [], [], []

    for L, ckpt_path in CKPT_DICT.items():

        print(f"[INFO] Latent size = {L}")

        sd = safe_load_state_dict(ckpt_path)

        model = build_model(L)
        model.load_state_dict(sd, strict=True)
        model.eval()

        mse0, mse_list, psnr0, psnr_list, ssim0, ssim_list = ablate_latent_dims(
            model, coords, target, mask, idx, H, W
        )

        xs = np.arange(cursor, cursor + len(mse_list))
        cursor += len(mse_list)

        mse_all.append((xs, mse_list, mse0))
        psnr_all.append((xs, psnr_list, psnr0))
        ssim_all.append((xs, ssim_list, ssim0))

    # ------------------- MSE -------------------
    plt.figure(figsize=(15,4))
    for xs, ys, o in mse_all:
        plt.plot(xs, ys, "b-")
        plt.plot(xs, [o]*len(xs), "g--")
    plt.xlabel("Ablated latent index")
    plt.ylabel("MSE")
    plt.savefig(out_dir / "Figure_MSE_combined.png", dpi=300)
    plt.close()

    # ------------------- PSNR -------------------
    plt.figure(figsize=(15,4))
    for xs, ys, o in psnr_all:
        plt.plot(xs, ys, "b-")
        plt.plot(xs, [o]*len(xs), "g--")
    plt.xlabel("Ablated latent index")
    plt.ylabel("PSNR")
    plt.savefig(out_dir / "Figure_PSNR_combined.png", dpi=300)
    plt.close()

    # ------------------- SSIM -------------------
    plt.figure(figsize=(15,4))
    for xs, ys, o in ssim_all:
        plt.plot(xs, ys, "b-")
        plt.plot(xs, [o]*len(xs), "g--")
    plt.xlabel("Ablated latent index")
    plt.ylabel("SSIM")
    plt.savefig(out_dir / "Figure_SSIM_combined.png", dpi=300)
    plt.close()

    print("[DONE] Ablation plots generated!")


if __name__ == "__main__":
    main()
