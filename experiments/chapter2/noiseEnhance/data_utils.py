"""Data loading and sparse observation utilities for chapter 2 experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FlowData:
    lats: np.ndarray
    lons: np.ndarray
    field: np.ndarray
    mask: np.ndarray
    meta: Any

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.field.shape


@dataclass(frozen=True)
class SparseObservation:
    timestep: int
    sampling_rate: float
    seed: int
    indices: np.ndarray
    coords: np.ndarray
    geo_coords: np.ndarray
    y_clean: np.ndarray
    frame_shape: tuple[int, int]


def load_flow_npz(path: str | Path) -> FlowData:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    dat = np.load(path, allow_pickle=True)
    required = {"lats", "lons", "temperature"}
    missing = sorted(required.difference(dat.files))
    if missing:
        raise KeyError(f"Missing keys in {path}: {missing}")

    lats = np.asarray(dat["lats"], dtype=np.float64)
    lons = np.asarray(dat["lons"], dtype=np.float64)
    field = np.asarray(dat["temperature"], dtype=np.float64)
    if field.ndim != 3:
        raise ValueError("temperature must have shape (T, H, W).")

    t_steps, height, width = field.shape
    if "mask" in dat.files:
        mask = np.asarray(dat["mask"]).astype(bool)
        if mask.ndim == 2:
            mask = np.broadcast_to(mask[None, :, :], (t_steps, height, width)).copy()
        elif mask.ndim == 3:
            if mask.shape[0] < t_steps:
                raise ValueError("mask has fewer timesteps than temperature.")
            mask = mask[:t_steps]
        else:
            raise ValueError("mask must have shape (H, W) or (T, H, W).")
    else:
        mask = np.ones((t_steps, height, width), dtype=bool)

    if mask.shape != field.shape:
        raise ValueError(f"mask shape {mask.shape} does not match field shape {field.shape}.")

    meta = dat["meta"] if "meta" in dat.files else None
    return FlowData(lats=lats, lons=lons, field=field, mask=mask, meta=meta)


def normalized_grid_coords(lats: np.ndarray, lons: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build local planar coordinates for graph distances.

    The graph uses an equirectangular approximation around the study-domain
    center instead of treating raw longitude and latitude degrees as equal
    units. Distances are expressed in kilometres; the graph kernels estimate
    their own scale from kNN distances.
    """
    earth_radius_km = 6371.0
    lat0 = np.deg2rad(float(np.nanmean(lats)))
    lon0 = np.deg2rad(float(np.nanmean(lons)))
    lat_rad = np.deg2rad(lats)
    lon_rad = np.deg2rad(lons)
    y_km = earth_radius_km * (lat_rad - lat0)
    x_km = earth_radius_km * np.cos(lat0) * (lon_rad - lon0)
    grid_y, grid_x = np.meshgrid(y_km, x_km, indexing="ij")
    grid_lat_geo, grid_lon_geo = np.meshgrid(lats, lons, indexing="ij")
    coords = np.stack([grid_y, grid_x], axis=-1)
    geo_coords = np.stack([grid_lat_geo, grid_lon_geo], axis=-1)
    return coords, geo_coords


def valid_flat_indices(mask_frame: np.ndarray) -> np.ndarray:
    valid = np.asarray(mask_frame).reshape(-1).astype(bool)
    indices = np.flatnonzero(valid)
    if indices.size == 0:
        raise ValueError("No valid pixels found in mask.")
    return indices


def sample_sparse_observation(
    data: FlowData,
    timestep: int,
    sampling_rate: float,
    seed: int,
) -> SparseObservation:
    t_steps, height, width = data.field.shape
    if timestep < 0 or timestep >= t_steps:
        raise ValueError(f"timestep={timestep} is out of range [0, {t_steps - 1}].")
    if sampling_rate <= 0:
        raise ValueError("sampling_rate must be positive.")

    valid = valid_flat_indices(data.mask[timestep])
    n_total = height * width
    n_points = int(round(float(sampling_rate) * n_total))
    n_points = max(1, min(n_points, valid.size))

    rng = np.random.default_rng(seed)
    chosen = rng.choice(valid, size=n_points, replace=False)
    chosen.sort()

    coords_grid, geo_grid = normalized_grid_coords(data.lats, data.lons)
    frame = data.field[timestep].reshape(-1)
    coords = coords_grid.reshape(-1, 2)[chosen]
    geo_coords = geo_grid.reshape(-1, 2)[chosen]
    y_clean = frame[chosen].astype(np.float64, copy=True)

    if not np.all(data.mask[timestep].reshape(-1)[chosen]):
        raise AssertionError("Sparse observation contains invalid masked points.")

    return SparseObservation(
        timestep=int(timestep),
        sampling_rate=float(sampling_rate),
        seed=int(seed),
        indices=chosen.astype(np.int64),
        coords=coords,
        geo_coords=geo_coords,
        y_clean=y_clean,
        frame_shape=(height, width),
    )


def sparse_to_grid(
    frame_shape: tuple[int, int],
    indices: np.ndarray,
    values: np.ndarray,
    fill_value: float = np.nan,
) -> np.ndarray:
    height, width = frame_shape
    grid = np.full(height * width, fill_value, dtype=np.float64)
    grid[np.asarray(indices, dtype=np.int64)] = np.asarray(values, dtype=np.float64)
    return grid.reshape(height, width)


def masked_frame(data: FlowData, timestep: int) -> np.ndarray:
    frame = data.field[timestep].astype(np.float64, copy=True)
    return np.where(data.mask[timestep], frame, np.nan)
