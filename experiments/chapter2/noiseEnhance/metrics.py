"""Metrics for sparse observation enhancement."""

from __future__ import annotations

import numpy as np


EPS = 1e-12


def compute_metrics(
    estimate: np.ndarray,
    target: np.ndarray,
    noisy_reference: np.ndarray | None = None,
) -> dict[str, float]:
    estimate = np.asarray(estimate, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    err = estimate - target
    mse = float(np.mean(err * err))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    data_range = float(np.max(target) - np.min(target))
    psnr = float(10.0 * np.log10((data_range * data_range + EPS) / (mse + EPS)))

    gain = 0.0
    if noisy_reference is not None:
        noisy_reference = np.asarray(noisy_reference, dtype=np.float64).reshape(-1)
        noisy_rmse = float(np.sqrt(np.mean((noisy_reference - target) ** 2)))
        gain = float((noisy_rmse - rmse) / max(noisy_rmse, EPS) * 100.0)

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "psnr": psnr,
        "gain_percent": gain,
    }

