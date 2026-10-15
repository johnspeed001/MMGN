"""Sparse observation enhancement methods for chapter 2."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .graphs import EPS, build_knn_graph, laplacian_from_weights, stable_solve


METHODS = [
    "noisy",
    "knn_mean",
    "idw",
    "gaussian_rbf",
    "median",
    "bilateral_nlm",
    "optimal_interp",
    "graph_laplacian",
    "graph_wiener",
    "graph_tv",
    "ragle",
    "amr_gtv",
]

METHOD_LABELS = {
    "noisy": "含噪观测",
    "knn_mean": "KNN均值",
    "idw": "反距离加权",
    "gaussian_rbf": "高斯核平滑",
    "median": "中值滤波",
    "bilateral_nlm": "双边非局部均值",
    "optimal_interp": "最优插值",
    "graph_laplacian": "图拉普拉斯",
    "graph_wiener": "图维纳滤波",
    "graph_tv": "图总变差",
    "ragle": "RAGLE基础版",
    "amr_gtv": "AMR-GTV",
}


def _as_1d(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values.reshape(-1)


def _neighbors_with_self(neighbors: np.ndarray) -> list[np.ndarray]:
    groups: list[np.ndarray] = []
    for idx in range(neighbors.shape[0]):
        groups.append(np.concatenate(([idx], neighbors[idx])))
    return groups


def local_median(values: np.ndarray, neighbors: np.ndarray) -> np.ndarray:
    values = _as_1d(values)
    med = np.empty_like(values)
    for idx, group in enumerate(_neighbors_with_self(neighbors)):
        med[idx] = np.median(values[group])
    return med


def local_mean(values: np.ndarray, neighbors: np.ndarray) -> np.ndarray:
    values = _as_1d(values)
    out = np.empty_like(values)
    for idx, group in enumerate(_neighbors_with_self(neighbors)):
        out[idx] = np.mean(values[group])
    return out


def local_confidence(values: np.ndarray, neighbors: np.ndarray, min_confidence: float = 0.05) -> np.ndarray:
    values = _as_1d(values)
    confidence = np.ones_like(values)
    for idx, group in enumerate(_neighbors_with_self(neighbors)):
        local = values[group]
        med = np.median(local)
        mad = np.median(np.abs(local - med)) + EPS
        residual = abs(values[idx] - med)
        confidence[idx] = 1.0 / (1.0 + (residual / mad) ** 2)
    return np.clip(confidence, float(min_confidence), 1.0)


def robust_scale(values: np.ndarray) -> float:
    values = _as_1d(values)
    if values.size == 0:
        return EPS
    med = float(np.median(values))
    return max(1.4826 * float(np.median(np.abs(values - med))), EPS)


def knn_mean(coords: np.ndarray, y_noisy: np.ndarray, k: int = 8) -> np.ndarray:
    graph = build_knn_graph(coords, k=k)
    return local_mean(y_noisy, graph.neighbors)


def median_filter(coords: np.ndarray, y_noisy: np.ndarray, k: int = 8) -> np.ndarray:
    graph = build_knn_graph(coords, k=k)
    return local_median(y_noisy, graph.neighbors)


def idw_smooth(coords: np.ndarray, y_noisy: np.ndarray, k: int = 8, power: float = 2.0) -> np.ndarray:
    y = _as_1d(y_noisy)
    graph = build_knn_graph(coords, k=k)
    if graph.neighbors.size == 0:
        return y.copy()
    row = np.arange(y.size)[:, None]
    dist = np.sqrt(np.maximum(graph.dist_sq[row, graph.neighbors], EPS))
    weights = 1.0 / np.maximum(dist, EPS) ** float(power)
    weights = weights / np.maximum(np.sum(weights, axis=1, keepdims=True), EPS)
    return np.sum(weights * y[graph.neighbors], axis=1)


def gaussian_rbf_smooth(coords: np.ndarray, y_noisy: np.ndarray, k: int = 8, bandwidth: float | None = None) -> np.ndarray:
    y = _as_1d(y_noisy)
    graph = build_knn_graph(coords, k=k)
    if graph.neighbors.size == 0:
        return y.copy()
    row = np.arange(y.size)[:, None]
    sigma = graph.tau if bandwidth is None else max(float(bandwidth), EPS)
    weights = np.exp(-graph.dist_sq[row, graph.neighbors] / (2.0 * sigma * sigma))
    weights = weights / np.maximum(np.sum(weights, axis=1, keepdims=True), EPS)
    return np.sum(weights * y[graph.neighbors], axis=1)


def bilateral_nlm(
    coords: np.ndarray,
    y_noisy: np.ndarray,
    k: int = 8,
    spatial_sigma: float | None = None,
    value_sigma: float | None = None,
) -> np.ndarray:
    y = _as_1d(y_noisy)
    graph = build_knn_graph(coords, k=k)
    if graph.neighbors.size == 0:
        return y.copy()
    row = np.arange(y.size)[:, None]
    sigma_x = graph.tau if spatial_sigma is None else max(float(spatial_sigma), EPS)
    sigma_y = robust_scale(y) if value_sigma is None else max(float(value_sigma), EPS)
    spatial = np.exp(-graph.dist_sq[row, graph.neighbors] / (2.0 * sigma_x * sigma_x))
    value_diff = y[row] - y[graph.neighbors]
    tonal = np.exp(-(value_diff * value_diff) / (2.0 * sigma_y * sigma_y))
    weights = spatial * tonal

    self_weight = np.ones((y.size, 1), dtype=np.float64)
    weights = np.concatenate([self_weight, weights], axis=1)
    values = np.concatenate([y[:, None], y[graph.neighbors]], axis=1)
    weights = weights / np.maximum(np.sum(weights, axis=1, keepdims=True), EPS)
    return np.sum(weights * values, axis=1)


def optimal_interpolation_prior(
    coords: np.ndarray,
    y_noisy: np.ndarray,
    noise_sigma: float | None = None,
    length_scale: float | None = None,
    background: float | None = None,
) -> np.ndarray:
    """Objective-analysis prior estimated from noisy sparse observations only."""
    y = _as_1d(y_noisy)
    if y.size <= 2:
        return y.copy()
    graph = build_knn_graph(coords, k=min(8, y.size - 1))
    dist_sq = graph.dist_sq
    length = max(float(length_scale), EPS) if length_scale is not None else max(2.0 * graph.tau, EPS)
    bg = float(np.median(y)) if background is None else float(background)
    sigma_b = max(robust_scale(y), EPS)
    sigma_o = max(float(noise_sigma), EPS) if noise_sigma is not None else max(estimate_global_noise_sigma(coords, y, k=graph.k), EPS)

    B = sigma_b * sigma_b * np.exp(-dist_sq / (2.0 * length * length))
    R = (sigma_o * sigma_o + EPS) * np.eye(y.size, dtype=np.float64)
    alpha = stable_solve(B + R, y - bg)
    return bg + B @ alpha


def confidence_from_noise_sigma(
    sigma_map: np.ndarray,
    residual: np.ndarray | None = None,
    min_confidence: float = 0.05,
) -> np.ndarray:
    sigma = np.maximum(_as_1d(sigma_map), EPS)
    sigma_med = max(float(np.median(sigma)), EPS)
    noise_conf = 1.0 / (1.0 + (sigma / (2.0 * sigma_med)) ** 2)
    if residual is None:
        return np.clip(noise_conf, float(min_confidence), 1.0)

    residual = np.abs(_as_1d(residual))
    score = residual / np.maximum(sigma, EPS)
    outlier_conf = 1.0 / (1.0 + (score / 3.0) ** 2)
    return np.clip(noise_conf * outlier_conf, float(min_confidence), 1.0)


def estimate_global_noise_sigma(coords: np.ndarray, y_noisy: np.ndarray, k: int = 8) -> float:
    y = _as_1d(y_noisy)
    if y.size <= 2:
        return max(float(np.std(y)) * 0.05, EPS)
    graph = build_knn_graph(coords, k=k)
    pilot = local_median(y, graph.neighbors)
    return min(robust_scale(y - pilot), max(float(np.std(y)), EPS))


def estimate_pointwise_noise(
    coords: np.ndarray,
    y_noisy: np.ndarray,
    k: int = 8,
    smooth_k: int | None = None,
    min_confidence: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate per-observation noise scale and confidence from noisy data only.

    The estimator first builds an edge-preserving pilot field, then estimates a
    robust local residual scale. No clean values or synthetic noise parameters
    are used, so it is suitable for real sparse observations.
    """
    y = _as_1d(y_noisy)
    if y.size <= 2:
        sigma = np.full_like(y, max(float(np.std(y)) * 0.05, EPS))
        return sigma, np.ones_like(y), y.copy()

    k_eff = max(2, min(int(k), y.size - 1))
    graph = build_knn_graph(coords, k=k_eff)
    median_pilot = local_median(y, graph.neighbors)
    mean_pilot = local_mean(y, graph.neighbors)
    pilot = 0.70 * median_pilot + 0.30 * mean_pilot
    residual = y - pilot

    local_sigma = np.empty_like(y)
    for idx, group in enumerate(_neighbors_with_self(graph.neighbors)):
        local_sigma[idx] = robust_scale(residual[group])
    local_sigma = np.maximum(local_sigma, 0.35 * np.abs(residual))

    smooth_k_eff = max(2, min(int(smooth_k or max(k_eff * 2, 12)), y.size - 1))
    smooth_graph = build_knn_graph(coords, k=smooth_k_eff)
    sigma_map = local_median(local_sigma, smooth_graph.neighbors)

    global_sigma = robust_scale(residual)
    y_std = max(float(np.std(y)), EPS)
    floor = max(0.005 * y_std, 0.20 * global_sigma, EPS)
    cap = max(6.0 * max(global_sigma, floor), 2.0 * floor)
    sigma_map = np.clip(sigma_map, floor, cap)
    confidence = confidence_from_noise_sigma(sigma_map, residual=residual, min_confidence=min_confidence)
    confidence = np.minimum(confidence, local_confidence(y, graph.neighbors, min_confidence=min_confidence))
    return sigma_map, confidence, pilot


def graph_laplacian(coords: np.ndarray, y_noisy: np.ndarray, k: int = 8, lam: float = 0.001) -> np.ndarray:
    y = _as_1d(y_noisy)
    if float(lam) == 0.0:
        return y.copy()
    graph = build_knn_graph(coords, k=k)
    A = np.eye(y.size, dtype=np.float64) + float(lam) * graph.L
    return stable_solve(A, y)


def graph_wiener(
    coords: np.ndarray,
    y_noisy: np.ndarray,
    k: int = 8,
    noise_sigma: float | None = None,
) -> np.ndarray:
    y = _as_1d(y_noisy)
    if y.size <= 2:
        return y.copy()
    graph = build_knn_graph(coords, k=k)
    eigvals, eigvecs = np.linalg.eigh(graph.L)
    coeff = eigvecs.T @ y
    noise_var = max((float(noise_sigma) if noise_sigma is not None else estimate_global_noise_sigma(coords, y, k=graph.k)) ** 2, EPS)
    signal_var = max(float(np.var(y)) - noise_var, EPS)
    cutoff = max(float(np.median(eigvals[eigvals > EPS])) if np.any(eigvals > EPS) else 1.0, EPS)
    response = signal_var / (signal_var + noise_var * (1.0 + eigvals / cutoff))
    return eigvecs @ (response * coeff)


def graph_tv(
    coords: np.ndarray,
    y_noisy: np.ndarray,
    k: int = 8,
    lam: float = 0.006,
    delta: float | None = None,
    n_iter: int = 6,
) -> np.ndarray:
    y = _as_1d(y_noisy)
    if float(lam) == 0.0:
        return y.copy()
    graph = build_knn_graph(coords, k=k)
    z = y.copy()
    tv_delta = float(delta) if delta is not None and delta > 0 else max(estimate_global_noise_sigma(coords, y, k=graph.k), float(np.std(y)) * 0.015, 1e-5)
    I = np.eye(y.size, dtype=np.float64)
    for _ in range(max(1, int(n_iter))):
        diff = z[:, None] - z[None, :]
        weights = graph.W / np.sqrt(diff * diff + tv_delta * tv_delta)
        np.fill_diagonal(weights, 0.0)
        L_tv = laplacian_from_weights(weights)
        z = stable_solve(I + float(lam) * L_tv, y)
    return z


def ragle(
    coords: np.ndarray,
    y_noisy: np.ndarray,
    k: int = 8,
    lam: float = 0.001,
    robust_confidence: bool = True,
    robust_edges: bool = True,
) -> np.ndarray:
    y = _as_1d(y_noisy)
    if float(lam) == 0.0:
        return y.copy()

    graph_base = build_knn_graph(coords, k=k)
    confidence = local_confidence(y, graph_base.neighbors) if robust_confidence else np.ones_like(y)
    graph = build_knn_graph(coords, k=k, tau=graph_base.tau, y_values=y if robust_edges else None)

    C = np.diag(confidence)
    A = C + float(lam) * graph.L
    b = confidence * y
    return stable_solve(A, b)


def amr_gtv(
    coords: np.ndarray,
    y_noisy: np.ndarray,
    ms_k: Iterable[int] = (4, 8, 16),
    lam: float = 0.006,
    noise_sigma: float | None = None,
    noise_sigma_map: np.ndarray | None = None,
    median_weight: float = 0.0,
    oi_gamma: float = 0.0,
    oi_length_scale: float | None = None,
    delta: float | None = None,
    n_iter: int = 6,
    use_multiscale: bool = True,
    use_adaptive: bool = True,
    use_oi_prior: bool = True,
) -> np.ndarray:
    """Adaptive multi-scale robust graph total variation.

    The quadratic system is solved by IRLS for a Charbonnier graph total-variation penalty.
    It is more edge-preserving than L2 Laplacian smoothing and adds a point-
    adaptive local-median prior for low-confidence observations.
    """
    y = _as_1d(y_noisy)
    lam = float(lam)
    eta = max(float(median_weight), 0.0)
    gamma = max(float(oi_gamma), 0.0)
    if lam == 0.0 and eta == 0.0 and gamma == 0.0:
        return y.copy()

    ms_k_list = [int(k) for k in ms_k]
    scale_ks = ms_k_list if use_multiscale else [ms_k_list[len(ms_k_list) // 2]]
    graphs = [build_knn_graph(coords, k=k) for k in scale_ks]
    base_graph = graphs[min(1, len(graphs) - 1)]

    pilot = y.copy()
    if use_adaptive:
        if noise_sigma_map is None:
            sigma_map, confidence, pilot = estimate_pointwise_noise(coords, y, k=base_graph.k)
        else:
            sigma_map = np.maximum(_as_1d(noise_sigma_map), EPS)
            residual = y - local_median(y, base_graph.neighbors)
            confidence = confidence_from_noise_sigma(sigma_map, residual=residual)
            confidence = np.minimum(confidence, local_confidence(y, base_graph.neighbors))
    else:
        sigma_scalar = float(noise_sigma) if noise_sigma is not None else estimate_global_noise_sigma(coords, y, k=base_graph.k)
        sigma_map = np.full_like(y, max(sigma_scalar, EPS))
        confidence = np.ones_like(y)

    sigma_scalar = float(noise_sigma) if noise_sigma is not None else float(np.median(sigma_map))
    z_median = local_median(y, base_graph.neighbors)
    z_oi = (
        optimal_interpolation_prior(coords, y, noise_sigma=sigma_scalar, length_scale=oi_length_scale)
        if use_oi_prior and gamma > 0.0
        else np.zeros_like(y)
    )
    z = 0.85 * y + 0.15 * pilot
    tv_delta = float(delta) if delta is not None and delta > 0 else max(sigma_scalar, float(np.median(sigma_map)), float(np.std(y)) * 0.015, 1e-5)
    C = np.diag(confidence)

    for _ in range(max(1, int(n_iter))):
        L_ms = np.zeros((y.size, y.size), dtype=np.float64)
        for graph in graphs:
            diff = z[:, None] - z[None, :]
            edge_gate = np.exp(-(diff * diff) / (2.0 * max(tv_delta * 4.0, EPS) ** 2)) if use_adaptive else 1.0
            irls = graph.W * edge_gate / np.sqrt(diff * diff + tv_delta * tv_delta)
            np.fill_diagonal(irls, 0.0)
            L_ms += laplacian_from_weights(irls) / len(graphs)

        # The median prior is point-adaptive: suspected noisy observations
        # borrow more from their local robust estimate, while high-confidence
        # observations remain dominated by data fidelity.
        median_confidence = eta * (1.0 - confidence)
        A = C + lam * L_ms + np.diag(median_confidence) + gamma * np.eye(y.size, dtype=np.float64)
        b = confidence * y + median_confidence * z_median + gamma * z_oi
        z = stable_solve(A, b)
    return z


def enhance_observation(
    coords: np.ndarray,
    y_noisy: np.ndarray,
    method: str,
    **kwargs,
) -> np.ndarray:
    method = method.lower()
    k = int(kwargs.get("k", 8))
    lam = float(kwargs.get("lam", kwargs.get("lambda_", 0.001)))
    ms_k = kwargs.get("ms_k", (4, 8, 16))
    noise_sigma = kwargs.get("noise_sigma", None)
    noise_sigma_map = kwargs.get("noise_sigma_map", None)
    y_std = max(float(np.std(_as_1d(y_noisy))), EPS)
    if noise_sigma is not None:
        noise_level = float(noise_sigma) / y_std
    elif noise_sigma_map is not None:
        noise_level = float(np.median(np.maximum(_as_1d(noise_sigma_map), EPS))) / y_std
    else:
        noise_level = estimate_global_noise_sigma(coords, y_noisy, k=k) / y_std
    default_amr_lam = 0.0 if lam == 0.0 else max(1e-5, 0.012 * noise_level * noise_level)
    default_median_weight = 0.0 if lam == 0.0 else min(1.20, max(0.0, 4.0 * noise_level - 0.10))
    default_oi_gamma = 0.0 if lam == 0.0 else min(0.20, max(0.0, 0.10 * noise_level))

    if method == "noisy":
        return _as_1d(y_noisy).copy()
    if method == "knn_mean":
        return knn_mean(coords, y_noisy, k=k)
    if method == "idw":
        return idw_smooth(coords, y_noisy, k=k, power=float(kwargs.get("idw_power", 2.0)))
    if method == "gaussian_rbf":
        return gaussian_rbf_smooth(coords, y_noisy, k=k, bandwidth=kwargs.get("rbf_bandwidth", None))
    if method == "median":
        return median_filter(coords, y_noisy, k=k)
    if method == "bilateral_nlm":
        return bilateral_nlm(
            coords,
            y_noisy,
            k=k,
            spatial_sigma=kwargs.get("bilateral_spatial_sigma", None),
            value_sigma=kwargs.get("bilateral_value_sigma", noise_sigma),
        )
    if method == "optimal_interp":
        return optimal_interpolation_prior(
            coords,
            y_noisy,
            noise_sigma=noise_sigma,
            length_scale=kwargs.get("oi_length_scale", None),
        )
    if method == "graph_laplacian":
        return graph_laplacian(coords, y_noisy, k=k, lam=lam)
    if method == "graph_wiener":
        return graph_wiener(coords, y_noisy, k=k, noise_sigma=noise_sigma)
    if method == "graph_tv":
        return graph_tv(
            coords,
            y_noisy,
            k=k,
            lam=float(kwargs.get("tv_lam", lam)),
            delta=kwargs.get("delta", None),
            n_iter=int(kwargs.get("n_iter", 6)),
        )
    if method == "ragle":
        return ragle(coords, y_noisy, k=k, lam=lam)
    if method == "amr_gtv":
        return amr_gtv(
            coords,
            y_noisy,
            ms_k=ms_k,
            lam=float(kwargs.get("amr_lam", default_amr_lam)),
            noise_sigma=noise_sigma,
            noise_sigma_map=noise_sigma_map,
            median_weight=float(kwargs.get("median_weight", default_median_weight)),
            oi_gamma=float(kwargs.get("oi_gamma", default_oi_gamma)),
            oi_length_scale=kwargs.get("oi_length_scale", None),
            delta=kwargs.get("delta", None),
            n_iter=int(kwargs.get("n_iter", 6)),
            use_multiscale=bool(kwargs.get("use_multiscale", True)),
            use_adaptive=bool(kwargs.get("use_adaptive", True)),
            use_oi_prior=bool(kwargs.get("use_oi_prior", True)),
        )
    raise ValueError(f"Unknown enhancement method: {method}")
