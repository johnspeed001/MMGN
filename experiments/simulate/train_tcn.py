# train_tcn.py
# Train TCN to predict delta, then xhat = y - delta.
# Supports ablation by zeroing attitude or drift channels at input.

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import os
import argparse
import json
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

if __package__:
    from .models import TCNRegressor
else:
    from models import TCNRegressor

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class WindowDataset(Dataset):
    """
    Loads segments from index.csv and samples random windows for training.
    Each item returns:
      inp:  (Cin, T)
      y_obs: (Cout, T)  (for reconstruction xhat = y - delta)
      x_true:(Cout, T)
    """
    def __init__(self, root_dir: str, index_df: pd.DataFrame,
                 win: int = 256, samples_per_seg: int = 40,
                 ablate: str = "none"):
        super().__init__()
        self.root_dir = root_dir
        self.df = index_df.reset_index(drop=True)
        self.win = win
        self.samples_per_seg = samples_per_seg
        self.ablate = ablate

        # Preload file list for speed
        self.files = self.df["file"].tolist()

    def __len__(self):
        return len(self.files) * self.samples_per_seg

    def _load_npz(self, fname: str) -> Dict[str, np.ndarray]:
        path = os.path.join(self.root_dir, fname)
        data = np.load(path, allow_pickle=True)
        return {k: data[k] for k in data.files}

    def __getitem__(self, idx: int):
        seg_idx = idx // self.samples_per_seg
        fname = self.files[seg_idx]
        d = self._load_npz(fname)

        x = d["x"].astype(np.float32)        # (N, 3)
        y = d["y"].astype(np.float32)        # (N, 3)
        att = d["att"].astype(np.float32)    # (N, 6)
        drift = d["drift"].astype(np.float32)# (N, 4)

        N = x.shape[0]
        if N <= self.win:
            start = 0
        else:
            start = np.random.randint(0, N - self.win)
        end = start + self.win

        xw = x[start:end]     # (T,3)
        yw = y[start:end]
        aw = att[start:end]
        pw = drift[start:end]

        # Build input channels: [y(3), att(6), drift(4)] => 13 channels
        inp = np.concatenate([yw, aw, pw], axis=1)  # (T,13)

        # Ablation
        # channels layout:
        # 0:2 y, 3:8 att, 9:12 drift
        if self.ablate == "no_att":
            inp[:, 3:9] = 0.0
        elif self.ablate == "no_drift":
            inp[:, 9:13] = 0.0
        elif self.ablate == "no_att_drift":
            inp[:, 3:13] = 0.0

        # to torch format (C,T)
        inp = torch.from_numpy(inp.T)   # (13,T)
        yw_t = torch.from_numpy(yw.T)   # (3,T)
        xw_t = torch.from_numpy(xw.T)   # (3,T)
        return inp, yw_t, xw_t

def mse_loss(pred, target):
    return (pred - target).pow(2).mean()

def smooth_l1_on_diff(xhat):
    # xhat: (B, C, T)
    diff = xhat[:, :, 1:] - xhat[:, :, :-1]
    return diff.abs().mean()

def run_epoch(model, loader, device, optimizer=None, lambda_smooth=0.05):
    train = optimizer is not None
    model.train(train)

    total = 0.0
    total_sup = 0.0
    total_sm = 0.0
    n = 0

    for inp, yobs, xtrue in loader:
        inp = inp.to(device)     # (B,Cin,T)
        yobs = yobs.to(device)   # (B,3,T)
        xtrue = xtrue.to(device) # (B,3,T)

        delta = model(inp)       # (B,3,T)
        xhat = yobs - delta

        sup = mse_loss(xhat, xtrue)
        sm = smooth_l1_on_diff(xhat)
        loss = sup + lambda_smooth * sm

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        total += float(loss.item())
        total_sup += float(sup.item())
        total_sm += float(sm.item())
        n += 1

    return {
        "loss": total / max(n, 1),
        "sup": total_sup / max(n, 1),
        "smooth": total_sm / max(n, 1),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default=str(PROJECT_ROOT / "data" / "simulated"))
    ap.add_argument("--win", type=int, default=256)
    ap.add_argument("--samples_per_seg", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda_smooth", type=float, default=0.05)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--ablate", type=str, default="none",
                    choices=["none", "no_att", "no_drift", "no_att_drift"])
    ap.add_argument("--out_dir", type=str, default=str(PROJECT_ROOT / "outputs" / "simulate" / "runs_tcn"))
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    idx_path = os.path.join(args.data_dir, "index.csv")
    if not os.path.exists(idx_path):
        raise FileNotFoundError(f"index.csv not found in {args.data_dir}. Run simulate_dataset.py first.")

    df = pd.read_csv(idx_path)
    df_train = df[df["split"] == "train"].copy()
    df_val = df[df["split"] == "val"].copy()

    train_ds = WindowDataset(args.data_dir, df_train, win=args.win,
                             samples_per_seg=args.samples_per_seg, ablate=args.ablate)
    val_ds = WindowDataset(args.data_dir, df_val, win=args.win,
                           samples_per_seg=max(10, args.samples_per_seg // 2), ablate=args.ablate)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=0, drop_last=False)

    cin = 13
    cout = 3
    model = TCNRegressor(cin=cin, cout=cout, channels=(64, 64, 64, 64), kernel_size=3, dropout=args.dropout)
    device = torch.device(args.device)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    best_path = os.path.join(args.out_dir, f"best_tcn_{args.ablate}.pth")
    log_path = os.path.join(args.out_dir, f"log_{args.ablate}.jsonl")

    cfg_save = vars(args).copy()
    cfg_save["cin"] = cin
    cfg_save["cout"] = cout
    with open(os.path.join(args.out_dir, f"train_cfg_{args.ablate}.json"), "w", encoding="utf-8") as f:
        json.dump(cfg_save, f, ensure_ascii=False, indent=2)

    patience = 10
    bad = 0

    for ep in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, device, optimizer=optimizer, lambda_smooth=args.lambda_smooth)
        va = run_epoch(model, val_loader, device, optimizer=None, lambda_smooth=args.lambda_smooth)

        rec = {"epoch": ep, "train": tr, "val": va}
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"[{ep:03d}] train loss={tr['loss']:.6f} (sup={tr['sup']:.6f}, sm={tr['smooth']:.6f}) | "
              f"val loss={va['loss']:.6f}")

        if va["loss"] < best_val - 1e-6:
            best_val = va["loss"]
            bad = 0
            torch.save({"model": model.state_dict(), "args": cfg_save}, best_path)
            print(f"  [SAVE] best -> {best_path} (val={best_val:.6f})")
        else:
            bad += 1
            if bad >= patience:
                print(f"[EARLY STOP] no improvement for {patience} epochs.")
                break

    print(f"[DONE] best val loss = {best_val:.6f}")
    print(f"Model saved at: {best_path}")

if __name__ == "__main__":
    main()