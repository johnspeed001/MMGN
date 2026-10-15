"""Shared plotting helpers for the six-method Chapter 2 workflow."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data_utils import FlowData, SparseObservation, masked_frame
from .methods import METHOD_LABELS, METHODS


PAPER_METHODS = list(METHODS)
PAPER_COLORS = {
    "noisy": "#4d4d4d",
    "knn_mean": "#9c9c9c",
    "idw": "#bdbdbd",
    "gaussian_rbf": "#8c8c8c",
    "median": "#7f7f7f",
    "bilateral_nlm": "#17becf",
    "optimal_interp": "#1f77b4",
    "graph_laplacian": "#9467bd",
    "graph_wiener": "#2ca02c",
    "graph_tv": "#8c564b",
    "ragle": "#ff7f0e",
    "amr_gtv": "#d62728",
}
PAPER_MARKERS = {
    "noisy": "o",
    "knn_mean": "v",
    "idw": "^",
    "gaussian_rbf": "<",
    "median": "P",
    "bilateral_nlm": "s",
    "optimal_interp": "h",
    "graph_laplacian": "D",
    "graph_wiener": "p",
    "graph_tv": "d",
    "ragle": "X",
    "amr_gtv": "*",
}


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _paper_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 140,
            "savefig.dpi": 320,
        }
    )


def save_fig(fig, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=320, bbox_inches="tight")
    plt.close(fig)


def _field_cmap():
    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad("lightgray")
    return cmap


def _imshow_field(ax, data: FlowData, timestep: int, title: str):
    frame = masked_frame(data, timestep)
    im = ax.imshow(
        frame,
        origin="lower",
        extent=[data.lons.min(), data.lons.max(), data.lats.min(), data.lats.max()],
        aspect="auto",
        cmap=_field_cmap(),
    )
    ax.set_title(title)
    ax.set_xlabel("经度")
    ax.set_ylabel("纬度")
    return im


def _scatter_values(
    ax,
    obs: SparseObservation,
    values: np.ndarray,
    title: str,
    cmap: str = "turbo",
    vmin: float | None = None,
    vmax: float | None = None,
    size: float = 12.0,
):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    im = ax.scatter(
        obs.geo_coords[:, 1],
        obs.geo_coords[:, 0],
        c=values,
        s=size,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=0,
    )
    ax.set_title(title)
    ax.set_xlabel("经度")
    ax.set_ylabel("纬度")
    return im


def plot_dataset_mask_sampling(data: FlowData, obs: SparseObservation, out_path: str | Path) -> None:
    _paper_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), constrained_layout=True)
    im0 = _imshow_field(axes[0], data, obs.timestep, "（a）完整流场")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.02)

    axes[1].imshow(
        data.mask[obs.timestep],
        origin="lower",
        extent=[data.lons.min(), data.lons.max(), data.lats.min(), data.lats.max()],
        aspect="auto",
        cmap="gray",
    )
    axes[1].set_title("（b）海域掩膜")
    axes[1].set_xlabel("经度")
    axes[1].set_ylabel("纬度")

    im2 = _imshow_field(axes[2], data, obs.timestep, "（c）稀疏观测点")
    axes[2].scatter(obs.geo_coords[:, 1], obs.geo_coords[:, 0], s=8, c="black", alpha=0.75)
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.02)
    save_fig(fig, out_path)


def plot_noise_model(obs: SparseObservation, y_noisy: np.ndarray, noise: np.ndarray, out_path: str | Path) -> None:
    _paper_style()
    vmin = float(np.min(obs.y_clean))
    vmax = float(np.max(obs.y_clean))
    vmax_noise = float(np.max(np.abs(noise))) if np.size(noise) else 1.0
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), constrained_layout=True)
    panels = [
        (obs.y_clean, "（a）真实观测值", "turbo", vmin, vmax),
        (y_noisy, "（b）含噪观测值", "turbo", vmin, vmax),
        (noise, "（c）噪声残差", "coolwarm", -vmax_noise, vmax_noise),
    ]
    for ax, (values, title, cmap, lo, hi) in zip(axes, panels):
        im = _scatter_values(ax, obs, values, title, cmap=cmap, vmin=lo, vmax=hi, size=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    save_fig(fig, out_path)


def plot_knn_graph(obs: SparseObservation, neighbors: np.ndarray, out_path: str | Path, max_nodes: int = 100) -> None:
    _paper_style()
    n_show = min(max_nodes, obs.coords.shape[0])
    subset = np.arange(n_show)
    subset_set = set(subset.tolist())
    fig, ax = plt.subplots(1, 1, figsize=(3.8, 3.3))
    for idx in subset:
        for nb in neighbors[idx]:
            if int(nb) in subset_set:
                ax.plot(
                    [obs.geo_coords[idx, 1], obs.geo_coords[nb, 1]],
                    [obs.geo_coords[idx, 0], obs.geo_coords[nb, 0]],
                    color="0.72",
                    linewidth=0.45,
                    zorder=1,
                )
    ax.scatter(obs.geo_coords[subset, 1], obs.geo_coords[subset, 0], s=18, c=obs.y_clean[subset], cmap="turbo", zorder=2)
    ax.set_title("局部kNN图结构")
    ax.set_xlabel("经度")
    ax.set_ylabel("纬度")
    save_fig(fig, out_path)


def method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method)
