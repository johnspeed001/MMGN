"""Noise models for sparse observations."""

from __future__ import annotations

import numpy as np


def add_relative_gaussian_noise(
    values: np.ndarray,
    noise_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    values = np.asarray(values, dtype=np.float64)
    if noise_ratio < 0:
        raise ValueError("noise_ratio must be non-negative.")

    std = float(np.std(values))
    sigma = float(noise_ratio) * std
    if sigma == 0.0:
        noise = np.zeros_like(values, dtype=np.float64)
    else:
        rng = np.random.default_rng(seed)
        noise = rng.normal(loc=0.0, scale=sigma, size=values.shape)
    return values + noise, noise, sigma


def add_observation_noise(
    values: np.ndarray,
    noise_ratio: float,
    seed: int,
    model: str = "gaussian",
    outlier_fraction: float = 0.05,
    outlier_scale: float = 6.0,
    return_sigma_map: bool = False,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray] | tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    """Add reproducible sensor noise.

    Parameters
    ----------
    values:
        Clean sparse observations.
    noise_ratio:
        Gaussian noise sigma as a fraction of the clean-observation standard
        deviation.
    model:
        ``gaussian``, ``gaussian_outlier``, ``heteroscedastic`` or
        ``heteroscedastic_outlier``. The heteroscedastic variants use a
        different Gaussian standard deviation at each observation point.

    Returns
    -------
    y_noisy, noise, sigma, outlier_mask. If ``return_sigma_map=True``, the
    per-point standard-deviation map is appended as the fifth return value.
    """
    values = np.asarray(values, dtype=np.float64)
    outlier_mask = np.zeros(values.shape, dtype=bool)
    if noise_ratio < 0:
        raise ValueError("noise_ratio must be non-negative.")

    base_sigma = float(noise_ratio) * float(np.std(values))
    rng = np.random.default_rng(seed)

    if model in {"gaussian", "gaussian_outlier"}:
        sigma_map = np.full(values.shape, base_sigma, dtype=np.float64)
        noise = np.zeros_like(values, dtype=np.float64) if base_sigma == 0.0 else rng.normal(loc=0.0, scale=base_sigma, size=values.shape)
        y_noisy = values + noise
    elif model in {"heteroscedastic", "heteroscedastic_outlier"}:
        if base_sigma == 0.0:
            sigma_map = np.zeros(values.shape, dtype=np.float64)
            noise = np.zeros_like(values, dtype=np.float64)
        else:
            raw_scale = rng.lognormal(mean=0.0, sigma=0.75, size=values.shape)
            raw_scale = np.clip(raw_scale, 0.25, 4.0)
            raw_scale = raw_scale / max(float(np.sqrt(np.mean(raw_scale * raw_scale))), 1e-12)
            sigma_map = base_sigma * raw_scale
            noise = rng.normal(loc=0.0, scale=sigma_map, size=values.shape)
        y_noisy = values + noise
    else:
        raise ValueError(f"Unknown noise model: {model}")

    if model in {"gaussian", "heteroscedastic"}:
        sigma_eff = float(np.sqrt(np.mean(sigma_map * sigma_map))) if sigma_map.size else 0.0
        if return_sigma_map:
            return y_noisy, noise, sigma_eff, outlier_mask, sigma_map
        return y_noisy, noise, sigma_eff, outlier_mask

    rng = np.random.default_rng(seed + 7919)
    n_outliers = int(round(float(outlier_fraction) * values.size))
    if n_outliers <= 0 or base_sigma == 0.0:
        sigma_eff = float(np.sqrt(np.mean(sigma_map * sigma_map))) if sigma_map.size else 0.0
        if return_sigma_map:
            return y_noisy, noise, sigma_eff, outlier_mask, sigma_map
        return y_noisy, noise, sigma_eff, outlier_mask

    idx = rng.choice(values.size, size=min(n_outliers, values.size), replace=False)
    impulse_scale = sigma_map[idx] if model == "heteroscedastic_outlier" else base_sigma
    impulse = rng.choice([-1.0, 1.0], size=idx.size) * float(outlier_scale) * impulse_scale
    y_noisy = y_noisy.copy()
    noise = noise.copy()
    y_noisy[idx] += impulse
    noise[idx] += impulse
    outlier_mask[idx] = True
    sigma_eff = float(np.sqrt(np.mean(sigma_map * sigma_map))) if sigma_map.size else 0.0
    if return_sigma_map:
        return y_noisy, noise, sigma_eff, outlier_mask, sigma_map
    return y_noisy, noise, sigma_eff, outlier_mask
