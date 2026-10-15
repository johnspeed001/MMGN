# simulate_dataset.py
# Pure simulation dataset for buoy disturbance compensation (single buoy)
# Generates paired (x_true, y_obs, attitude, drift) sequences under 16 conditions.

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import os
import json
import math
import random
import argparse
from dataclasses import asdict, dataclass
import numpy as np
import pandas as pd

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def band_limited_signal(t, freqs, amps, phases):
    # sum_i amp_i * sin(2π f_i t + phase_i)
    sig = np.zeros_like(t, dtype=np.float64)
    for f, a, ph in zip(freqs, amps, phases):
        sig += a * np.sin(2 * np.pi * f * t + ph)
    return sig

def smooth_lowfreq_noise(n, std=1.0, win=301):
    # low-frequency noise by smoothing white noise
    x = np.random.randn(n) * std
    win = int(win)
    if win < 3:
        return x
    if win % 2 == 0:
        win += 1
    k = np.ones(win, dtype=np.float64) / win
    return np.convolve(x, k, mode="same")

def random_walk(n, step_std=1e-4):
    # cumulative random walk
    steps = np.random.randn(n) * step_std
    return np.cumsum(steps)

def rms_to_scale(x, target_rms):
    rms = np.sqrt(np.mean(x**2)) + 1e-12
    return x * (target_rms / rms)

def rotation_matrix_yaw_pitch_roll(yaw, pitch, roll):
    # ZYX: Rz(yaw)*Ry(pitch)*Rx(roll)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    Rz = np.array([[cy, -sy, 0],
                   [sy,  cy, 0],
                   [ 0,   0, 1]], dtype=np.float64)
    Ry = np.array([[ cp, 0, sp],
                   [  0, 1,  0],
                   [-sp, 0, cp]], dtype=np.float64)
    Rx = np.array([[1,  0,   0],
                   [0, cr, -sr],
                   [0, sr,  cr]], dtype=np.float64)
    return Rz @ Ry @ Rx

@dataclass
class SimConfig:
    fs: float = 10.0
    minutes: float = 10.0
    # wave truth
    wave_m: float = 1.2             # scale for z(t)
    wave_freq_min: float = 0.05
    wave_freq_max: float = 0.5
    wave_components: int = 8
    # current truth
    current_base: float = 0.5       # m/s baseline magnitude
    current_var: float = 0.2        # variability
    current_lowfreq_win: int = 2001
    # attitude
    att_main_freq_min: float = 0.05
    att_main_freq_max: float = 0.5
    att_components: int = 5
    yaw_lowfreq_win: int = 5001
    # drift
    drift_lowfreq_win: int = 4001
    gps_noise_std_m: float = 2.0
    # sensor noises
    z_noise_std: float = 0.02
    uv_noise_std: float = 0.03
    # bias drift (random walk)
    z_bias_step: float = 2e-5
    uv_bias_step: float = 2e-5
    # spikes
    spike_prob_per_sample: float = 1e-4
    spike_sigma_mult: float = 6.0
    # platform velocity coupling
    alpha_platform: float = 0.8

def generate_truth(cfg: SimConfig, n: int, t: np.ndarray):
    # z(t): wave-like
    freqs = np.random.uniform(cfg.wave_freq_min, cfg.wave_freq_max, size=cfg.wave_components)
    amps  = np.random.uniform(0.2, 1.0, size=cfg.wave_components)
    phases= np.random.uniform(0, 2*np.pi, size=cfg.wave_components)
    z = band_limited_signal(t, freqs, amps, phases)
    z = rms_to_scale(z, target_rms=cfg.wave_m)

    # currents u(t), v(t): low-frequency smooth + mild high-freq
    u_lf = smooth_lowfreq_noise(n, std=1.0, win=cfg.current_lowfreq_win)
    v_lf = smooth_lowfreq_noise(n, std=1.0, win=cfg.current_lowfreq_win)
    u_lf = rms_to_scale(u_lf, target_rms=cfg.current_var)
    v_lf = rms_to_scale(v_lf, target_rms=cfg.current_var)

    theta = np.random.uniform(0, 2*np.pi)
    u0 = cfg.current_base * np.cos(theta)
    v0 = cfg.current_base * np.sin(theta)

    u = u0 + u_lf
    v = v0 + v_lf
    return z.astype(np.float32), u.astype(np.float32), v.astype(np.float32)

def generate_attitude(cfg: SimConfig, n: int, t: np.ndarray, roll_rms_deg: float, pitch_rms_deg: float):
    # roll/pitch: band-limited random sum + a bit of lowfreq noise
    freqs_r = np.random.uniform(cfg.att_main_freq_min, cfg.att_main_freq_max, size=cfg.att_components)
    amps_r  = np.random.uniform(0.2, 1.0, size=cfg.att_components)
    phases_r= np.random.uniform(0, 2*np.pi, size=cfg.att_components)
    roll = band_limited_signal(t, freqs_r, amps_r, phases_r) + 0.2 * smooth_lowfreq_noise(n, std=1.0, win=2001)

    freqs_p = np.random.uniform(cfg.att_main_freq_min, cfg.att_main_freq_max, size=cfg.att_components)
    amps_p  = np.random.uniform(0.2, 1.0, size=cfg.att_components)
    phases_p= np.random.uniform(0, 2*np.pi, size=cfg.att_components)
    pitch = band_limited_signal(t, freqs_p, amps_p, phases_p) + 0.2 * smooth_lowfreq_noise(n, std=1.0, win=2001)

    roll = rms_to_scale(roll, target_rms=math.radians(roll_rms_deg))
    pitch = rms_to_scale(pitch, target_rms=math.radians(pitch_rms_deg))

    # yaw: low frequency drift-like
    yaw = smooth_lowfreq_noise(n, std=1.0, win=cfg.yaw_lowfreq_win)
    yaw = rms_to_scale(yaw, target_rms=math.radians(15.0))  # ~15deg RMS heading wander

    # gyro (angular rates) approximate by finite diff
    dt = 1.0 / cfg.fs
    roll_rate  = np.gradient(roll, dt)
    pitch_rate = np.gradient(pitch, dt)
    yaw_rate   = np.gradient(yaw, dt)

    return (roll.astype(np.float32), pitch.astype(np.float32), yaw.astype(np.float32),
            roll_rate.astype(np.float32), pitch_rate.astype(np.float32), yaw_rate.astype(np.float32))

def generate_drift(cfg: SimConfig, n: int):
    # drift velocity: low-frequency random vector with target magnitude scale
    vx = smooth_lowfreq_noise(n, std=1.0, win=cfg.drift_lowfreq_win)
    vy = smooth_lowfreq_noise(n, std=1.0, win=cfg.drift_lowfreq_win)
    return vx.astype(np.float32), vy.astype(np.float32)

def inject_spikes(x, prob, sigma_mult, base_std):
    # add spikes with certain probability
    n = x.shape[0]
    mask = (np.random.rand(n) < prob).astype(np.float32)
    spikes = np.random.randn(n).astype(np.float32) * (sigma_mult * base_std)
    return x + mask * spikes

def simulate_one_segment(cfg: SimConfig, roll_rms_deg: float, drift_speed: float):
    n = int(cfg.minutes * 60 * cfg.fs)
    dt = 1.0 / cfg.fs
    t = np.arange(n, dtype=np.float64) * dt

    # truth
    z, u, v = generate_truth(cfg, n, t)

    # attitude
    roll, pitch, yaw, gr, gp, gy = generate_attitude(cfg, n, t, roll_rms_deg, roll_rms_deg)

    # drift velocity (raw shape), then scale to target speed
    vx0, vy0 = generate_drift(cfg, n)
    speed0 = np.sqrt(vx0**2 + vy0**2) + 1e-8
    # normalize then scale to drift_speed (RMS-ish)
    vx = (vx0 / np.mean(speed0)) * drift_speed
    vy = (vy0 / np.mean(speed0)) * drift_speed

    # positions by integration + GPS noise
    px = np.cumsum(vx) * dt
    py = np.cumsum(vy) * dt
    px_gps = px + np.random.randn(n).astype(np.float32) * cfg.gps_noise_std_m
    py_gps = py + np.random.randn(n).astype(np.float32) * cfg.gps_noise_std_m

    # sensor bias drifts (random walk)
    bz = random_walk(n, step_std=cfg.z_bias_step).astype(np.float32)
    bu = random_walk(n, step_std=cfg.uv_bias_step).astype(np.float32)
    bv = random_walk(n, step_std=cfg.uv_bias_step).astype(np.float32)

    # Observation model
    # z channel affected by projection + bias + noise
    yz = z * (np.cos(roll) * np.cos(pitch)).astype(np.float32) + bz + np.random.randn(n).astype(np.float32) * cfg.z_noise_std
    yz = inject_spikes(yz, cfg.spike_prob_per_sample, cfg.spike_sigma_mult, cfg.z_noise_std)

    # currents affected by rotation + platform velocity coupling + bias + noise
    yu = np.zeros(n, dtype=np.float32)
    yv = np.zeros(n, dtype=np.float32)
    noise_u = np.random.randn(n).astype(np.float32) * cfg.uv_noise_std
    noise_v = np.random.randn(n).astype(np.float32) * cfg.uv_noise_std

    for i in range(n):
        R = rotation_matrix_yaw_pitch_roll(float(yaw[i]), float(pitch[i]), float(roll[i]))
        vec = np.array([float(u[i]), float(v[i]), 0.0], dtype=np.float64)
        vec_r = R @ vec
        # platform velocity coupling (use drift v)
        yu[i] = float(vec_r[0]) + cfg.alpha_platform * float(vx[i]) + float(bu[i]) + float(noise_u[i])
        yv[i] = float(vec_r[1]) + cfg.alpha_platform * float(vy[i]) + float(bv[i]) + float(noise_v[i])

    yu = inject_spikes(yu, cfg.spike_prob_per_sample, cfg.spike_sigma_mult, cfg.uv_noise_std)
    yv = inject_spikes(yv, cfg.spike_prob_per_sample, cfg.spike_sigma_mult, cfg.uv_noise_std)

    # pack arrays
    x = np.stack([z, u, v], axis=1).astype(np.float32)                 # (n, 3)
    y = np.stack([yz, yu, yv], axis=1).astype(np.float32)              # (n, 3)
    att = np.stack([roll, pitch, yaw, gr, gp, gy], axis=1).astype(np.float32)  # (n, 6)
    drift = np.stack([px_gps, py_gps, vx, vy], axis=1).astype(np.float32)      # (n, 4)

    return x, y, att, drift

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default=str(PROJECT_ROOT / "data" / "simulated"))
    ap.add_argument("--fs", type=float, default=10.0)
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--train_per_cond", type=int, default=8)
    ap.add_argument("--val_per_cond", type=int, default=2)
    ap.add_argument("--test_per_cond", type=int, default=2)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--make_generalization_split", action="store_true",
                    help="train only on L1/L2/L3 x D0/D1/D2, test includes L4 or D3")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = SimConfig(fs=args.fs, minutes=args.minutes)
    cfg_dict = asdict(cfg)
    with open(os.path.join(args.out_dir, "sim_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg_dict, f, ensure_ascii=False, indent=2)

    att_levels_deg = [1.0, 3.0, 6.0, 10.0]      # L1..L4
    drift_levels = [0.0, 0.1, 0.3, 0.5]         # D0..D3

    rows = []
    idx = 0

    for li, att_deg in enumerate(att_levels_deg, start=1):
        for di, drift_speed in enumerate(drift_levels, start=0):
            cond_name = f"L{li}_D{di}"

            # decide split counts
            n_train = args.train_per_cond
            n_val = args.val_per_cond
            n_test = args.test_per_cond

            # if generalization split: move "hard" conditions (L4 or D3) mostly to test
            if args.make_generalization_split:
                is_hard = (li == 4) or (di == 3)
                if is_hard:
                    n_train = max(1, args.train_per_cond // 4)
                    n_val = max(1, args.val_per_cond // 2)
                    n_test = args.test_per_cond + args.train_per_cond  # more test
                else:
                    # easy/mid: keep as usual
                    pass

            # create segments
            for split, count in [("train", n_train), ("val", n_val), ("test", n_test)]:
                for _ in range(count):
                    x, y, att, drift = simulate_one_segment(cfg, att_deg, drift_speed)
                    fn = f"seg_{idx:06d}_{cond_name}_{split}.npz"
                    fp = os.path.join(args.out_dir, fn)
                    np.savez_compressed(fp, x=x, y=y, att=att, drift=drift,
                                        cond=cond_name, split=split,
                                        att_deg=att_deg, drift_speed=drift_speed,
                                        fs=args.fs)
                    rows.append({
                        "id": idx,
                        "file": fn,
                        "cond": cond_name,
                        "split": split,
                        "att_level": li,
                        "att_rms_deg": att_deg,
                        "drift_level": di,
                        "drift_speed": drift_speed,
                        "fs": args.fs,
                        "minutes": args.minutes
                    })
                    idx += 1

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, "index.csv"), index=False, encoding="utf-8-sig")
    print(f"[OK] wrote {len(df)} segments to {args.out_dir}")
    print(df.groupby(["split"]).size())

if __name__ == "__main__":
    main()