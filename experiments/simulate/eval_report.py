# eval_report.py
# Evaluate trained TCN on full sequences, produce metrics tables and plots.

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import os
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch

if __package__:
    from .models import TCNRegressor
else:
    from models import TCNRegressor

def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))

def mae(a, b):
    return float(np.mean(np.abs(a - b)))

def lowpass_filter(x, win=31):
    # simple moving average, x: (N,C)
    if win <= 1:
        return x
    k = np.ones(win, dtype=np.float64) / win
    y = np.zeros_like(x, dtype=np.float64)
    for c in range(x.shape[1]):
        y[:, c] = np.convolve(x[:, c], k, mode="same")
    return y.astype(np.float32)

def build_input(y, att, drift, ablate="none"):
    inp = np.concatenate([y, att, drift], axis=1).astype(np.float32)  # (N,13)
    if ablate == "no_att":
        inp[:, 3:9] = 0.0
    elif ablate == "no_drift":
        inp[:, 9:13] = 0.0
    elif ablate == "no_att_drift":
        inp[:, 3:13] = 0.0
    return inp

@torch.no_grad()
def run_model_on_sequence(model, inp_np, device):
    # inp_np: (N,13)
    x = torch.from_numpy(inp_np.T[None, ...]).to(device)  # (1,13,N)
    delta = model(x).cpu().numpy()[0].T  # (N,3)
    return delta.astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default=str(PROJECT_ROOT / "data" / "simulated"))
    ap.add_argument("--ckpt", type=str, default=str(PROJECT_ROOT / "outputs" / "simulate" / "runs_tcn" / "best_tcn_none.pth"))
    ap.add_argument("--ablate", type=str, default="none",
                    choices=["none", "no_att", "no_drift", "no_att_drift"])
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--filter_win", type=int, default=31)
    ap.add_argument("--out_dir", type=str, default=str(PROJECT_ROOT / "outputs" / "simulate" / "eval_out"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    idx_path = os.path.join(args.data_dir, "index.csv")
    df = pd.read_csv(idx_path)

    # Load model
    ckpt = torch.load(args.ckpt, map_location="cpu")
    train_args = ckpt.get("args", {})
    cin = int(train_args.get("cin", 13))
    cout = int(train_args.get("cout", 3))

    model = TCNRegressor(cin=cin, cout=cout, channels=(64,64,64,64), kernel_size=3, dropout=float(train_args.get("dropout", 0.1)))
    model.load_state_dict(ckpt["model"], strict=True)
    device = torch.device(args.device)
    model.to(device)
    model.eval()

    records = []
    hardest = None
    hardest_key = None

    # Evaluate per segment
    for _, row in df.iterrows():
        fpath = os.path.join(args.data_dir, row["file"])
        d = np.load(fpath, allow_pickle=True)
        x = d["x"].astype(np.float32)        # (N,3)
        y = d["y"].astype(np.float32)
        att = d["att"].astype(np.float32)
        drift = d["drift"].astype(np.float32)

        # Baselines
        y_f = lowpass_filter(y, win=args.filter_win)  # just filtering y as naive baseline

        # DL
        inp = build_input(y, att, drift, ablate=args.ablate)
        delta = run_model_on_sequence(model, inp, device)  # (N,3)
        xhat = y - delta

        # Metrics
        rec = {
            "id": int(row["id"]),
            "file": row["file"],
            "split": row["split"],
            "cond": row["cond"],
            "att_level": int(row["att_level"]),
            "drift_level": int(row["drift_level"]),
            "att_rms_deg": float(row["att_rms_deg"]),
            "drift_speed": float(row["drift_speed"]),
        }

        for ci, name in enumerate(["z", "u", "v"]):
            rec[f"raw_rmse_{name}"] = rmse(y[:,ci], x[:,ci])
            rec[f"raw_mae_{name}"]  = mae(y[:,ci], x[:,ci])

            rec[f"filt_rmse_{name}"] = rmse(y_f[:,ci], x[:,ci])
            rec[f"filt_mae_{name}"]  = mae(y_f[:,ci], x[:,ci])

            rec[f"dl_rmse_{name}"] = rmse(xhat[:,ci], x[:,ci])
            rec[f"dl_mae_{name}"]  = mae(xhat[:,ci], x[:,ci])

        records.append(rec)

        # Track hardest (L4_D3 preferred)
        key = (int(row["att_level"]), int(row["drift_level"]), str(row["split"]))
        if hardest is None:
            hardest = (x, y, y_f, xhat, row)
            hardest_key = key
        else:
            # prioritize att_level=4 and drift_level=3 in test split
            cur_score = (int(row["split"]=="test"), int(row["att_level"]), int(row["drift_level"]))
            best_score = (int(hardest[4]["split"]=="test"), int(hardest[4]["att_level"]), int(hardest[4]["drift_level"]))
            if cur_score > best_score:
                hardest = (x, y, y_f, xhat, row)
                hardest_key = key

    out_csv = os.path.join(args.out_dir, f"metrics_{args.ablate}.csv")
    mdf = pd.DataFrame(records)
    mdf.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] metrics saved: {out_csv}")

    # Summary tables by split/cond
    def summarize(split: str):
        sdf = mdf[mdf["split"] == split]
        agg = {}
        for name in ["z","u","v"]:
            agg[f"raw_rmse_{name}"] = sdf[f"raw_rmse_{name}"].mean()
            agg[f"filt_rmse_{name}"] = sdf[f"filt_rmse_{name}"].mean()
            agg[f"dl_rmse_{name}"] = sdf[f"dl_rmse_{name}"].mean()
        return agg

    summary = {"train": summarize("train"), "val": summarize("val"), "test": summarize("test")}
    with open(os.path.join(args.out_dir, f"summary_{args.ablate}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[OK] summary saved: {os.path.join(args.out_dir, f'summary_{args.ablate}.json')}")

    # Plot RMSE vs attitude level (test split)
    test = mdf[mdf["split"]=="test"].copy()

    def plot_curve(xcol, title, fname):
        plt.figure()
        for method in ["raw", "filt", "dl"]:
            ys = []
            xs = sorted(test[xcol].unique().tolist())
            for xv in xs:
                part = test[test[xcol]==xv]
                ys.append(part[[f"{method}_rmse_z", f"{method}_rmse_u", f"{method}_rmse_v"]].mean().mean())
            plt.plot(xs, ys, marker="o", label=method)
        plt.xlabel(xcol)
        plt.ylabel("RMSE (avg over z/u/v)")
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, fname), dpi=200)
        plt.close()

    plot_curve("att_level", f"RMSE vs Attitude Level (ablate={args.ablate})", f"rmse_vs_att_{args.ablate}.png")
    plot_curve("drift_level", f"RMSE vs Drift Level (ablate={args.ablate})", f"rmse_vs_drift_{args.ablate}.png")
    print(f"[OK] curves saved in {args.out_dir}")

    # Plot hardest case time series (z channel)
    x, y, y_f, xhat, row = hardest
    plt.figure(figsize=(10,4))
    n = x.shape[0]
    tt = np.arange(n) / float(row["fs"])
    # plot first 1200 samples (~120s at 10Hz) for readability
    L = min(n, 1200)
    plt.plot(tt[:L], x[:L,0], label="true z")
    plt.plot(tt[:L], y[:L,0], label="obs z")
    plt.plot(tt[:L], y_f[:L,0], label="filter z")
    plt.plot(tt[:L], xhat[:L,0], label="dl zhat")
    plt.xlabel("time (s)")
    plt.ylabel("z")
    plt.title(f"Hard Case: {row['cond']} split={row['split']} (ablate={args.ablate})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, f"hardcase_z_{args.ablate}.png"), dpi=200)
    plt.close()
    print(f"[OK] hardcase plot saved.")

if __name__ == "__main__":
    main()