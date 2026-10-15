"""kNN graph and graph Laplacian builders."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


EPS = 1e-12


@dataclass(frozen=True)
class KNNGraph:
    coords: np.ndarray
    k: int
    neighbors: np.ndarray
    dist_sq: np.ndarray
    W: np.ndarray
    L: np.ndarray
    tau: float


def pairwise_squared_distances(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2:
        raise ValueError("coords must have shape (N, D).")
    diff = coords[:, None, :] - coords[None, :, :]
    dist_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dist_sq, 0.0)
    return dist_sq


def knn_indices(dist_sq: np.ndarray, k: int) -> np.ndarray:
    dist_sq = np.asarray(dist_sq, dtype=np.float64)
    if dist_sq.ndim != 2 or dist_sq.shape[0] != dist_sq.shape[1]:
        raise ValueError("dist_sq must be a square distance matrix.")

    n_points = dist_sq.shape[0]
    if n_points <= 1:
        return np.zeros((n_points, 0), dtype=np.int64)
    k_eff = max(1, min(int(k), n_points - 1))
    work = dist_sq.copy()
    np.fill_diagonal(work, np.inf)
    part = np.argpartition(work, kth=k_eff - 1, axis=1)[:, :k_eff]
    row = np.arange(n_points)[:, None]
    order = np.argsort(work[row, part], axis=1)
    return part[row, order].astype(np.int64)


def estimate_tau(dist_sq: np.ndarray, neighbors: np.ndarray) -> float:
    if neighbors.size == 0:
        return 1.0
    row = np.arange(neighbors.shape[0])[:, None]
    d = np.sqrt(np.maximum(dist_sq[row, neighbors], 0.0))
    tau = float(np.median(d[d > 0])) if np.any(d > 0) else 1.0
    return max(tau, EPS)


def laplacian_from_weights(W: np.ndarray) -> np.ndarray:
    degree = np.sum(W, axis=1)
    return np.diag(degree) - W


def robust_beta_from_edges(values: np.ndarray, neighbors: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if neighbors.size == 0:
        return float(np.std(values) + EPS)
    row = np.arange(neighbors.shape[0])[:, None]
    diffs = np.abs(values[row] - values[neighbors]).reshape(-1)
    beta = float(np.median(diffs[diffs > 0])) if np.any(diffs > 0) else float(np.std(values))
    return max(beta, EPS)


def build_knn_graph(
    coords: np.ndarray,
    k: int = 8,
    tau: float | None = None,
    y_values: np.ndarray | None = None,
    robust_beta: float | None = None,
) -> KNNGraph:
    coords = np.asarray(coords, dtype=np.float64)
    n_points = coords.shape[0]
    dist_sq = pairwise_squared_distances(coords)
    neighbors = knn_indices(dist_sq, k)
    tau_eff = estimate_tau(dist_sq, neighbors) if tau is None else max(float(tau), EPS)

    W = np.zeros((n_points, n_points), dtype=np.float64)
    if neighbors.size:
        row = np.arange(n_points)[:, None]
        edge_dist_sq = dist_sq[row, neighbors]
        weights = np.exp(-edge_dist_sq / (2.0 * tau_eff * tau_eff))

        if y_values is not None:
            y = np.asarray(y_values, dtype=np.float64)
            beta = robust_beta_from_edges(y, neighbors) if robust_beta is None else max(float(robust_beta), EPS)
            value_diff = y[row] - y[neighbors]
            robust = np.exp(-(value_diff * value_diff) / (2.0 * beta * beta))
            weights = weights * robust

        W[row, neighbors] = weights

    W = np.maximum(W, W.T)
    np.fill_diagonal(W, 0.0)
    L = laplacian_from_weights(W)
    return KNNGraph(coords=coords, k=neighbors.shape[1] if neighbors.ndim == 2 else 0, neighbors=neighbors, dist_sq=dist_sq, W=W, L=L, tau=tau_eff)


def stable_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        jitter = 1e-8 * np.eye(A.shape[0], dtype=np.float64)
        try:
            return np.linalg.solve(A + jitter, b)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(A + jitter, b, rcond=None)[0]
