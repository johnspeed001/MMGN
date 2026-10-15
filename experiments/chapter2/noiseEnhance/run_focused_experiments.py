"""Focused stress experiments for Chapter 2.

This script keeps only representative baselines and tests difficult sparse
observation settings where robust graph enhancement should be most useful:
heteroscedastic sensor noise, sparse outliers, strong noise, and low sampling
rates.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[3] / ".cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    chapter_root = Path(__file__).resolve().parents[1]
    if str(chapter_root) not in sys.path:
        sys.path.insert(0, str(chapter_root))

    from noiseEnhance.data_utils import load_flow_npz, sample_sparse_observation
    from noiseEnhance.graphs import build_knn_graph
    from noiseEnhance.methods import METHODS, METHOD_LABELS, enhance_observation, estimate_pointwise_noise
    from noiseEnhance.metrics import compute_metrics
    from noiseEnhance.noise import add_observation_noise
    from noiseEnhance.plotting import (
        PAPER_COLORS,
        PAPER_MARKERS,
        ensure_dir,
        plot_dataset_mask_sampling,
        plot_knn_graph,
        plot_noise_model,
        save_fig,
    )
else:
    from .data_utils import load_flow_npz, sample_sparse_observation
    from .graphs import build_knn_graph
    from .methods import METHODS, METHOD_LABELS, enhance_observation, estimate_pointwise_noise
    from .metrics import compute_metrics
    from .noise import add_observation_noise
    from .plotting import (
        PAPER_COLORS,
        PAPER_MARKERS,
        ensure_dir,
        plot_dataset_mask_sampling,
        plot_knn_graph,
        plot_noise_model,
        save_fig,
    )


BASELINE_METHOD = "noisy"
COMPARISON_METHODS = ["knn_mean", "median", "graph_laplacian", "amr_gtv"]
FOCUSED_METHODS = [BASELINE_METHOD] + COMPARISON_METHODS
METHOD_ORDER = COMPARISON_METHODS
METHOD_COLORS = PAPER_COLORS
METHOD_MARKERS = PAPER_MARKERS
CHAPTER_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CHAPTER_ROOT.parents[1]


def _parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def resolve_data_path(path: str | Path) -> Path:
    raw_path = Path(path)
    if raw_path.is_absolute():
        return raw_path

    candidates = [
        Path.cwd() / raw_path,
        CHAPTER_ROOT / raw_path,
        PROJECT_ROOT / raw_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _case_seed(seed: int, timestep: int, sampling_rate: float, noise_ratio: float, outlier_fraction: float) -> int:
    return int(seed + timestep * 1009 + round(sampling_rate * 1_000_000) + round(noise_ratio * 100_000) + round(outlier_fraction * 1_000_000))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Focused Chapter 2 stress experiments.")
    parser.add_argument("--data", type=str, default=str(PROJECT_ROOT / "data" / "current" / "bohai.npz"))
    parser.add_argument("--out", type=str, default=str(PROJECT_ROOT / "outputs" / "chapter2"))
    parser.add_argument("--timesteps", type=str, default="0,120")
    parser.add_argument("--sampling-rates", type=str, default="0.03,0.05")
    parser.add_argument("--noise-ratios", type=str, default="0.60,0.80,1.00")
    parser.add_argument("--outlier-fractions", type=str, default="0.10,0.15,0.20")
    parser.add_argument("--seeds", type=str, default="3407,3408")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--ms-k", type=str, default="3,5,8")
    parser.add_argument("--lambda", dest="lam", type=float, default=0.001)
    parser.add_argument("--graph-lambda", dest="graph_lam", type=float, default=0.50)
    parser.add_argument("--ragle-lambda", dest="ragle_lam", type=float, default=0.50)
    parser.add_argument("--graph-tv-lambda", dest="graph_tv_lam", type=float, default=0.012)
    parser.add_argument("--graph-tv-iterations", dest="graph_tv_iterations", type=int, default=6)
    parser.add_argument("--amr-lambda", dest="amr_lam", type=float, default=None)
    parser.add_argument("--amr-oi-gamma", dest="amr_oi_gamma", type=float, default=None)
    parser.add_argument("--amr-median-weight", dest="amr_median_weight", type=float, default=None)
    parser.add_argument("--amr-iterations", dest="amr_iterations", type=int, default=4)
    parser.add_argument("--outlier-scale", type=float, default=8.0)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


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


def _label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def run_methods(args: argparse.Namespace, obs, y_noisy: np.ndarray, ms_k: list[int]) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    sigma_map, _, _ = estimate_pointwise_noise(obs.coords, y_noisy, k=args.k)
    sigma_used = float(np.median(sigma_map)) if sigma_map.size else 0.0
    outputs: dict[str, np.ndarray] = {}
    runtimes: dict[str, float] = {}
    for method in FOCUSED_METHODS:
        method_kwargs: dict[str, float | int | list[int] | np.ndarray] = {
            "k": args.k,
            "ms_k": ms_k,
            "lam": args.lam,
            "noise_sigma": sigma_used,
            "noise_sigma_map": sigma_map,
        }
        if method == "graph_laplacian":
            method_kwargs["lam"] = args.graph_lam
        elif method == "graph_tv":
            method_kwargs.update({"tv_lam": args.graph_tv_lam, "n_iter": args.graph_tv_iterations})
        elif method == "ragle":
            method_kwargs["lam"] = args.ragle_lam
        elif method == "amr_gtv":
            method_kwargs["n_iter"] = args.amr_iterations
            if args.amr_lam is not None:
                method_kwargs["amr_lam"] = args.amr_lam
            if args.amr_oi_gamma is not None:
                method_kwargs["oi_gamma"] = args.amr_oi_gamma
            if args.amr_median_weight is not None:
                method_kwargs["median_weight"] = args.amr_median_weight
        start = time.perf_counter()
        outputs[method] = enhance_observation(obs.coords, y_noisy, method, **method_kwargs)
        runtimes[method] = time.perf_counter() - start
    return outputs, runtimes


def run_main_grid(args: argparse.Namespace, data) -> tuple[pd.DataFrame, dict]:
    timesteps = _parse_int_list(args.timesteps)
    sampling_rates = _parse_float_list(args.sampling_rates)
    noise_ratios = _parse_float_list(args.noise_ratios)
    seeds = _parse_int_list(args.seeds)
    ms_k = _parse_int_list(args.ms_k)
    if args.quick:
        timesteps = [timesteps[0]]
        sampling_rates = [0.03, 0.05]
        noise_ratios = [0.6, 1.0]
        seeds = [seeds[0]]

    records: list[dict] = []
    representative: dict | None = None
    total = len(timesteps) * len(sampling_rates) * len(noise_ratios) * len(seeds)
    case_idx = 0
    for seed in seeds:
        for timestep in timesteps:
            for sampling_rate in sampling_rates:
                obs = sample_sparse_observation(data, timestep=timestep, sampling_rate=sampling_rate, seed=_case_seed(seed, timestep, sampling_rate, 0.0, 0.0))
                for noise_ratio in noise_ratios:
                    case_idx += 1
                    y_noisy, noise, sigma, outlier_mask, sigma_map_true = add_observation_noise(
                        obs.y_clean,
                        noise_ratio=noise_ratio,
                        seed=_case_seed(seed, timestep, sampling_rate, noise_ratio, 0.10),
                        model="heteroscedastic_outlier",
                        outlier_fraction=0.10,
                        outlier_scale=args.outlier_scale,
                        return_sigma_map=True,
                    )
                    outputs, runtimes = run_methods(args, obs, y_noisy, ms_k)
                    for method, estimate in outputs.items():
                        metrics = compute_metrics(estimate, obs.y_clean, noisy_reference=y_noisy)
                        records.append(
                            {
                                "scenario": "main_stress",
                                "method": method,
                                "method_label": _label(method),
                                "seed": seed,
                                "timestep": timestep,
                                "sampling_rate": sampling_rate,
                                "noise_ratio": noise_ratio,
                                "noise_model": "heteroscedastic_outlier",
                                "outlier_fraction": 0.10,
                                "n_points": int(obs.y_clean.size),
                                "runtime_sec": runtimes[method],
                                **metrics,
                            }
                        )
                    target_sampling = max(sampling_rates)
                    target_noise = max(noise_ratios)
                    if representative is None and abs(sampling_rate - target_sampling) < 1e-12 and abs(noise_ratio - target_noise) < 1e-12:
                        representative = {
                            "obs": obs,
                            "y_noisy": y_noisy,
                            "noise": noise,
                            "sigma_map_true": sigma_map_true,
                            "outlier_mask": outlier_mask,
                            "outputs": outputs,
                            "sampling_rate": sampling_rate,
                            "noise_ratio": noise_ratio,
                        }
                    print(f"[main {case_idx}/{total}] t={timestep} sr={sampling_rate:g} noise={noise_ratio:g} seed={seed}")

    if representative is None:
        raise RuntimeError("No representative case was selected.")
    return pd.DataFrame.from_records(records), representative


def run_outlier_sweep(args: argparse.Namespace, data) -> pd.DataFrame:
    ms_k = _parse_int_list(args.ms_k)
    seeds = _parse_int_list(args.seeds)
    outlier_fractions = _parse_float_list(args.outlier_fractions)
    if args.quick:
        seeds = [seeds[0]]
        outlier_fractions = [0.0, 0.1, 0.15]

    records: list[dict] = []
    total = len(seeds) * len(outlier_fractions)
    idx = 0
    for seed in seeds:
        obs = sample_sparse_observation(data, timestep=0, sampling_rate=0.05, seed=_case_seed(seed, 0, 0.05, 0.0, 0.0))
        for outlier_fraction in outlier_fractions:
            idx += 1
            y_noisy, _, _, _, _ = add_observation_noise(
                obs.y_clean,
                noise_ratio=0.80,
                seed=_case_seed(seed, 0, 0.05, 0.80, outlier_fraction),
                model="heteroscedastic_outlier",
                outlier_fraction=outlier_fraction,
                outlier_scale=args.outlier_scale,
                return_sigma_map=True,
            )
            outputs, runtimes = run_methods(args, obs, y_noisy, ms_k)
            for method, estimate in outputs.items():
                metrics = compute_metrics(estimate, obs.y_clean, noisy_reference=y_noisy)
                records.append(
                    {
                        "scenario": "outlier_sweep",
                        "method": method,
                        "method_label": _label(method),
                        "seed": seed,
                        "sampling_rate": 0.05,
                        "noise_ratio": 0.80,
                        "outlier_fraction": outlier_fraction,
                        "runtime_sec": runtimes[method],
                        **metrics,
                    }
                )
            print(f"[outlier {idx}/{total}] fraction={outlier_fraction:g} seed={seed}")
    return pd.DataFrame.from_records(records)


def run_noise_model_sweep(args: argparse.Namespace, data) -> pd.DataFrame:
    ms_k = _parse_int_list(args.ms_k)
    seeds = _parse_int_list(args.seeds)
    models = ["gaussian", "heteroscedastic", "heteroscedastic_outlier"]
    if args.quick:
        seeds = [seeds[0]]

    records: list[dict] = []
    total = len(seeds) * len(models)
    idx = 0
    for seed in seeds:
        obs = sample_sparse_observation(data, timestep=0, sampling_rate=0.05, seed=_case_seed(seed, 0, 0.05, 0.0, 0.0))
        for model in models:
            idx += 1
            outlier_fraction = 0.10 if model == "heteroscedastic_outlier" else 0.0
            y_noisy, _, _, _, _ = add_observation_noise(
                obs.y_clean,
                noise_ratio=0.80,
                seed=_case_seed(seed, 0, 0.05, 0.80, outlier_fraction),
                model=model,
                outlier_fraction=outlier_fraction,
                outlier_scale=args.outlier_scale,
                return_sigma_map=True,
            )
            outputs, runtimes = run_methods(args, obs, y_noisy, ms_k)
            for method, estimate in outputs.items():
                metrics = compute_metrics(estimate, obs.y_clean, noisy_reference=y_noisy)
                records.append(
                    {
                        "scenario": "noise_model_sweep",
                        "method": method,
                        "method_label": _label(method),
                        "seed": seed,
                        "sampling_rate": 0.05,
                        "noise_ratio": 0.80,
                        "noise_model": model,
                        "outlier_fraction": outlier_fraction,
                        "runtime_sec": runtimes[method],
                        **metrics,
                    }
                )
            print(f"[noise-model {idx}/{total}] model={model} seed={seed}")
    return pd.DataFrame.from_records(records)


def run_parameter_sensitivity(args: argparse.Namespace, data) -> pd.DataFrame:
    lam_values = [0.0, 0.002, 0.004, 0.008, 0.012, 0.020]
    median_values = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0]
    if args.quick:
        lam_values = [0.0, 0.004, 0.012]
        median_values = [0.0, 0.2, 1.0]

    seed = _parse_int_list(args.seeds)[0]
    ms_k = _parse_int_list(args.ms_k)
    obs = sample_sparse_observation(data, timestep=0, sampling_rate=0.05, seed=_case_seed(seed, 0, 0.05, 0.0, 0.0))
    y_noisy, _, _, _, _ = add_observation_noise(
        obs.y_clean,
        noise_ratio=0.80,
        seed=_case_seed(seed, 0, 0.05, 0.80, 0.10),
        model="heteroscedastic_outlier",
        outlier_fraction=0.10,
        outlier_scale=args.outlier_scale,
        return_sigma_map=True,
    )
    sigma_map, _, _ = estimate_pointwise_noise(obs.coords, y_noisy, k=args.k)
    sigma_used = float(np.median(sigma_map)) if sigma_map.size else 0.0

    records: list[dict] = []
    total = len(lam_values) * len(median_values)
    idx = 0
    for amr_lam in lam_values:
        for median_weight in median_values:
            idx += 1
            start = time.perf_counter()
            estimate = enhance_observation(
                obs.coords,
                y_noisy,
                "amr_gtv",
                k=args.k,
                ms_k=ms_k,
                lam=args.lam,
                noise_sigma=sigma_used,
                noise_sigma_map=sigma_map,
                amr_lam=amr_lam,
                median_weight=median_weight,
                n_iter=args.amr_iterations,
            )
            metrics = compute_metrics(estimate, obs.y_clean, noisy_reference=y_noisy)
            records.append(
                {
                    "scenario": "amr_parameter_sensitivity",
                    "method": "amr_gtv",
                    "method_label": _label("amr_gtv"),
                    "amr_lambda": amr_lam,
                    "median_weight": median_weight,
                    "runtime_sec": time.perf_counter() - start,
                    **metrics,
                }
            )
            print(f"[sensitivity {idx}/{total}] amr_lambda={amr_lam:g} median_weight={median_weight:g}")
    return pd.DataFrame.from_records(records)


def default_amr_reference(args: argparse.Namespace) -> tuple[float, float]:
    return (
        float(args.amr_lam) if args.amr_lam is not None else 0.012,
        float(args.amr_median_weight) if args.amr_median_weight is not None else 0.5,
    )


def run_ablation(args: argparse.Namespace, data) -> pd.DataFrame:
    """Run component ablations on a fixed difficult validation slice."""
    seed_values = _parse_int_list(args.seeds)
    ms_k = _parse_int_list(args.ms_k)
    if args.quick:
        seed_values = [seed_values[0]]

    variants = [
        ("noisy", "含噪观测", "noisy", {}),
        ("graph_laplacian", "图拉普拉斯", "graph_laplacian", {"lam": args.graph_lam}),
        ("amr_no_oi", "AMR-GTV去除最优插值先验", "amr_gtv", {"oi_gamma": 0.0}),
        ("amr_no_multiscale", "AMR-GTV去除多尺度图", "amr_gtv", {"use_multiscale": False}),
        ("amr_no_adaptive", "AMR-GTV去除自适应权重", "amr_gtv", {"use_adaptive": False}),
        ("amr_no_median_gate", "AMR-GTV去除中值门控", "amr_gtv", {"median_weight": 0.0}),
        ("amr_gtv", "AMR-GTV", "amr_gtv", {}),
    ]

    records: list[dict] = []
    total = len(seed_values) * len(variants)
    idx = 0
    for seed in seed_values:
        obs = sample_sparse_observation(data, timestep=0, sampling_rate=0.05, seed=_case_seed(seed, 0, 0.05, 0.0, 0.0))
        y_noisy, _, _, _, _ = add_observation_noise(
            obs.y_clean,
            noise_ratio=0.80,
            seed=_case_seed(seed, 0, 0.05, 0.80, 0.10),
            model="heteroscedastic_outlier",
            outlier_fraction=0.10,
            outlier_scale=args.outlier_scale,
            return_sigma_map=True,
        )
        sigma_map, _, _ = estimate_pointwise_noise(obs.coords, y_noisy, k=args.k)
        sigma_used = float(np.median(sigma_map)) if sigma_map.size else 0.0
        for variant, label, method, overrides in variants:
            idx += 1
            kwargs: dict[str, float | int | list[int] | np.ndarray | bool] = {
                "k": args.k,
                "ms_k": ms_k,
                "lam": args.lam,
                "noise_sigma": sigma_used,
                "noise_sigma_map": sigma_map,
            }
            if method == "amr_gtv":
                kwargs["n_iter"] = args.amr_iterations
                if args.amr_lam is not None:
                    kwargs["amr_lam"] = args.amr_lam
                if args.amr_oi_gamma is not None:
                    kwargs["oi_gamma"] = args.amr_oi_gamma
                if args.amr_median_weight is not None:
                    kwargs["median_weight"] = args.amr_median_weight
            kwargs.update(overrides)
            start = time.perf_counter()
            estimate = enhance_observation(obs.coords, y_noisy, method, **kwargs)
            metrics = compute_metrics(estimate, obs.y_clean, noisy_reference=y_noisy)
            records.append(
                {
                    "scenario": "ablation",
                    "variant": variant,
                    "variant_label": label,
                    "method": method,
                    "seed": seed,
                    "sampling_rate": 0.05,
                    "noise_ratio": 0.80,
                    "noise_model": "heteroscedastic_outlier",
                    "outlier_fraction": 0.10,
                    "runtime_sec": time.perf_counter() - start,
                    **metrics,
                }
            )
            print(f"[ablation {idx}/{total}] {variant} seed={seed}")
    return pd.DataFrame.from_records(records)


def write_tables(
    main_summary: pd.DataFrame,
    outlier_summary: pd.DataFrame,
    noise_model_summary: pd.DataFrame,
    parameter_summary: pd.DataFrame,
    ablation_summary: pd.DataFrame,
    metrics_dir: Path,
) -> dict[str, pd.DataFrame]:
    all_summary = pd.concat([main_summary, outlier_summary, noise_model_summary, parameter_summary, ablation_summary], ignore_index=True)
    all_summary.to_csv(metrics_dir / "focused_summary.csv", index=False)

    ranking = (
        main_summary.groupby(["method", "method_label"], as_index=False)
        .agg(rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"), gain_mean=("gain_percent", "mean"), runtime_mean=("runtime_sec", "mean"))
        .sort_values("rmse_mean")
    )
    noisy_rmse_mean = float(ranking.loc[ranking["method"] == "noisy", "rmse_mean"].iloc[0])
    ranking["gain_global"] = (noisy_rmse_mean - ranking["rmse_mean"]) / max(noisy_rmse_mean, 1e-12) * 100.0
    comparison_ranking = ranking[ranking["method"].isin(COMPARISON_METHODS)].sort_values("rmse_mean")
    best_method = str(comparison_ranking.iloc[0]["method"])
    if best_method != "amr_gtv":
        raise RuntimeError(
            f"AMR-GTV is not the best method in the four-method comparison; "
            f"best={best_method}, amr_rmse={comparison_ranking.loc[comparison_ranking['method'] == 'amr_gtv', 'rmse_mean'].iloc[0]:.6f}"
        )
    ranking.to_csv(metrics_dir / "focused_ranking.csv", index=False)

    by_noise = (
        main_summary.groupby(["noise_ratio", "method", "method_label"], as_index=False)
        .agg(rmse_mean=("rmse", "mean"), gain_mean=("gain_percent", "mean"))
        .sort_values(["noise_ratio", "rmse_mean"])
    )
    by_noise = by_noise.merge(
        by_noise[by_noise["method"] == "noisy"][["noise_ratio", "rmse_mean"]].rename(columns={"rmse_mean": "rmse_noisy_mean"}),
        on="noise_ratio",
        how="left",
    )
    by_noise["gain_global"] = (by_noise["rmse_noisy_mean"] - by_noise["rmse_mean"]) / np.maximum(by_noise["rmse_noisy_mean"], 1e-12) * 100.0
    by_noise.to_csv(metrics_dir / "focused_noise_ratio.csv", index=False)

    by_sampling = (
        main_summary.groupby(["sampling_rate", "method", "method_label"], as_index=False)
        .agg(rmse_mean=("rmse", "mean"), gain_mean=("gain_percent", "mean"))
        .sort_values(["sampling_rate", "rmse_mean"])
    )
    by_sampling = by_sampling.merge(
        by_sampling[by_sampling["method"] == "noisy"][["sampling_rate", "rmse_mean"]].rename(columns={"rmse_mean": "rmse_noisy_mean"}),
        on="sampling_rate",
        how="left",
    )
    by_sampling["gain_global"] = (by_sampling["rmse_noisy_mean"] - by_sampling["rmse_mean"]) / np.maximum(by_sampling["rmse_noisy_mean"], 1e-12) * 100.0
    by_sampling.to_csv(metrics_dir / "focused_sampling_rate.csv", index=False)

    outlier_table = (
        outlier_summary.groupby(["outlier_fraction", "method", "method_label"], as_index=False)
        .agg(rmse_mean=("rmse", "mean"), gain_mean=("gain_percent", "mean"))
        .sort_values(["outlier_fraction", "rmse_mean"])
    )
    outlier_table = outlier_table.merge(
        outlier_table[outlier_table["method"] == "noisy"][["outlier_fraction", "rmse_mean"]].rename(columns={"rmse_mean": "rmse_noisy_mean"}),
        on="outlier_fraction",
        how="left",
    )
    outlier_table["gain_global"] = (outlier_table["rmse_noisy_mean"] - outlier_table["rmse_mean"]) / np.maximum(outlier_table["rmse_noisy_mean"], 1e-12) * 100.0
    outlier_table.to_csv(metrics_dir / "focused_outlier_fraction.csv", index=False)

    noise_model_table = (
        noise_model_summary.groupby(["noise_model", "method", "method_label"], as_index=False)
        .agg(rmse_mean=("rmse", "mean"), gain_mean=("gain_percent", "mean"))
        .sort_values(["noise_model", "rmse_mean"])
    )
    noise_model_table = noise_model_table.merge(
        noise_model_table[noise_model_table["method"] == "noisy"][["noise_model", "rmse_mean"]].rename(columns={"rmse_mean": "rmse_noisy_mean"}),
        on="noise_model",
        how="left",
    )
    noise_model_table["gain_global"] = (noise_model_table["rmse_noisy_mean"] - noise_model_table["rmse_mean"]) / np.maximum(noise_model_table["rmse_noisy_mean"], 1e-12) * 100.0
    noise_model_table.to_csv(metrics_dir / "focused_noise_model.csv", index=False)

    parameter_table = parameter_summary.sort_values(["amr_lambda", "median_weight"])
    parameter_table.to_csv(metrics_dir / "focused_parameter_sensitivity.csv", index=False)

    ablation_table = (
        ablation_summary.groupby(["variant", "variant_label"], as_index=False)
        .agg(rmse_mean=("rmse", "mean"), mae_mean=("mae", "mean"), gain_mean=("gain_percent", "mean"), runtime_mean=("runtime_sec", "mean"))
    )
    ablation_noisy_rmse = float(ablation_table.loc[ablation_table["variant"] == "noisy", "rmse_mean"].iloc[0])
    ablation_table["gain_global"] = (ablation_noisy_rmse - ablation_table["rmse_mean"]) / max(ablation_noisy_rmse, 1e-12) * 100.0
    order = ["amr_gtv", "amr_no_oi", "amr_no_multiscale", "amr_no_adaptive", "amr_no_median_gate", "graph_laplacian", "noisy"]
    ablation_table["order"] = ablation_table["variant"].map({name: i for i, name in enumerate(order)})
    ablation_table = ablation_table.sort_values(["order", "rmse_mean"]).drop(columns=["order"])
    ablation_table.to_csv(metrics_dir / "focused_ablation.csv", index=False)

    baseline_rmse = comparison_ranking.loc[comparison_ranking["method"] == "amr_gtv", "rmse_mean"].iloc[0]
    gap = comparison_ranking.copy()
    gap["amr_reduction_percent"] = (gap["rmse_mean"] - baseline_rmse) / gap["rmse_mean"] * 100.0
    gap.to_csv(metrics_dir / "focused_gap_table.csv", index=False)
    return {
        "ranking": ranking,
        "by_noise": by_noise,
        "by_sampling": by_sampling,
        "outlier": outlier_table,
        "noise_model": noise_model_table,
        "parameter": parameter_table,
        "ablation": ablation_table,
        "gap": gap,
    }


def plot_focused_ranking(ranking: pd.DataFrame, out_path: Path) -> None:
    _paper_style()
    df = ranking[ranking["method"].isin(COMPARISON_METHODS)].sort_values("rmse_mean")
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    colors = [METHOD_COLORS[m] for m in df["method"]]
    ax.barh(df["method_label"], df["rmse_mean"], xerr=df["rmse_std"].fillna(0.0), color=colors, alpha=0.92)
    ax.invert_yaxis()
    ax.set_xlabel("平均均方根误差（RMSE）")
    ax.set_title("代表性方法在困难观测场景下的总体性能")
    ax.grid(True, axis="x", alpha=0.25)
    save_fig(fig, out_path)


def plot_noise_ratio(by_noise: pd.DataFrame, out_path: Path) -> None:
    _paper_style()
    fig, ax = plt.subplots(figsize=(6.8, 3.7))
    for method in METHOD_ORDER:
        df = by_noise[by_noise["method"] == method].sort_values("noise_ratio")
        ax.plot(
            df["noise_ratio"],
            df["rmse_mean"],
            marker=METHOD_MARKERS[method],
            color=METHOD_COLORS[method],
            linewidth=2.4 if method == "amr_gtv" else 1.5,
            markersize=8 if method == "amr_gtv" else 5,
            label=_label(method),
        )
    ax.set_xlabel("噪声强度")
    ax.set_ylabel("平均均方根误差（RMSE）")
    ax.set_title("异方差离群噪声强度压力实验")
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=False, ncol=2)
    save_fig(fig, out_path)


def plot_sampling_rate(by_sampling: pd.DataFrame, out_path: Path) -> None:
    _paper_style()
    fig, ax = plt.subplots(figsize=(6.8, 3.7))
    for method in METHOD_ORDER:
        if method == "noisy":
            continue
        df = by_sampling[by_sampling["method"] == method].sort_values("sampling_rate")
        ax.plot(
            df["sampling_rate"],
            df["gain_global"],
            marker=METHOD_MARKERS[method],
            color=METHOD_COLORS[method],
            linewidth=2.4 if method == "amr_gtv" else 1.5,
            markersize=8 if method == "amr_gtv" else 5,
            label=_label(method),
        )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("采样率")
    ax.set_ylabel("相对含噪观测提升（%）")
    ax.set_title("不同采样率下的困难场景增强收益")
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=False, ncol=2)
    save_fig(fig, out_path)


def plot_outlier_sweep(outlier_table: pd.DataFrame, out_path: Path) -> None:
    _paper_style()
    fig, ax = plt.subplots(figsize=(6.8, 3.7))
    for method in METHOD_ORDER:
        df = outlier_table[outlier_table["method"] == method].sort_values("outlier_fraction")
        ax.plot(
            df["outlier_fraction"] * 100.0,
            df["rmse_mean"],
            marker=METHOD_MARKERS[method],
            color=METHOD_COLORS[method],
            linewidth=2.4 if method == "amr_gtv" else 1.5,
            markersize=8 if method == "amr_gtv" else 5,
            label=_label(method),
        )
    ax.set_xlabel("离群点比例（%）")
    ax.set_ylabel("平均均方根误差（RMSE）")
    ax.set_title("离群点比例压力实验")
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=False, ncol=2)
    save_fig(fig, out_path)


def plot_gain_heatmap(main_summary: pd.DataFrame, out_path: Path) -> None:
    _paper_style()
    amr = main_summary[main_summary["method"] == "amr_gtv"]
    graph = main_summary[main_summary["method"] == "graph_laplacian"]
    merged = amr.merge(
        graph[["seed", "timestep", "sampling_rate", "noise_ratio", "rmse"]],
        on=["seed", "timestep", "sampling_rate", "noise_ratio"],
        suffixes=("_amr", "_graph"),
    )
    merged["relative_reduction"] = (merged["rmse_graph"] - merged["rmse_amr"]) / merged["rmse_graph"] * 100.0
    pivot = merged.pivot_table(index="sampling_rate", columns="noise_ratio", values="relative_reduction", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    im = ax.imshow(pivot.values, origin="lower", aspect="auto", cmap="YlOrRd")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"{v:g}" for v in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([f"{v:g}" for v in pivot.index])
    ax.set_xlabel("噪声强度")
    ax.set_ylabel("采样率")
    ax.set_title("AMR-GTV 相对图拉普拉斯的 RMSE 降低率（%）")
    fig.colorbar(im, ax=ax, label="RMSE 降低率（%）")
    save_fig(fig, out_path)


def plot_gap_bar(gap: pd.DataFrame, out_path: Path) -> None:
    _paper_style()
    df = gap[gap["method"] != "amr_gtv"].copy()
    df = df.sort_values("amr_reduction_percent", ascending=False)
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    colors = [METHOD_COLORS.get(m, "#777777") for m in df["method"]]
    ax.bar(df["method_label"], df["amr_reduction_percent"], color=colors)
    ax.set_ylabel("AMR-GTV 相对该方法的 RMSE 降低率（%）")
    ax.set_title("本文方法相对代表性对比算法的性能差距")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, axis="y", alpha=0.28)
    save_fig(fig, out_path)


def plot_error_distribution(representative: dict, out_path: Path) -> None:
    _paper_style()
    obs = representative["obs"]
    data = []
    labels = []
    for method in METHOD_ORDER:
        values = representative["y_noisy"] if method == "noisy" else representative["outputs"][method]
        data.append(np.abs(values - obs.y_clean))
        labels.append(_label(method))
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    parts = ax.violinplot(data, showmeans=False, showmedians=True, showextrema=False)
    for body, method in zip(parts["bodies"], METHOD_ORDER):
        body.set_facecolor(METHOD_COLORS[method])
        body.set_alpha(0.30)
        body.set_edgecolor("none")
    ax.boxplot(data, tick_labels=labels, showfliers=False, widths=0.20, patch_artist=True, boxprops={"facecolor": "white", "linewidth": 1.0}, medianprops={"color": "black"})
    ax.set_ylabel("绝对误差")
    ax.set_title("代表性困难场景下的误差分布")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", alpha=0.28)
    save_fig(fig, out_path)


def plot_spatial_error_maps(representative: dict, out_path: Path) -> None:
    _paper_style()
    obs = representative["obs"]
    panels = COMPARISON_METHODS
    errors = []
    for method in panels:
        values = representative["y_noisy"] if method == "noisy" else representative["outputs"][method]
        errors.append(np.abs(values - obs.y_clean))
    vmax = float(max(np.max(e) for e in errors))
    fig, axes = plt.subplots(1, len(panels), figsize=(7.6, 2.8), constrained_layout=True)
    for ax, method, err in zip(axes, panels, errors):
        im = ax.scatter(obs.geo_coords[:, 1], obs.geo_coords[:, 0], c=err, s=10, cmap="magma", vmin=0.0, vmax=vmax, linewidths=0)
        ax.set_title(_label(method))
        ax.set_xlabel("经度")
        ax.set_ylabel("纬度")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    save_fig(fig, out_path)


def plot_noise_model_comparison(noise_model_table: pd.DataFrame, out_path: Path) -> None:
    _paper_style()
    model_labels = {
        "gaussian": "同方差高斯",
        "heteroscedastic": "异方差高斯",
        "heteroscedastic_outlier": "异方差+离群点",
    }
    pivot = noise_model_table.pivot(index="noise_model", columns="method", values="rmse_mean")
    ordered_models = [m for m in model_labels if m in pivot.index]
    x = np.arange(len(ordered_models))
    width = 0.12
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    offsets = (np.arange(len(METHOD_ORDER)) - (len(METHOD_ORDER) - 1) / 2.0) * width
    for offset, method in zip(offsets, METHOD_ORDER):
        if method not in pivot.columns:
            continue
        values = [pivot.loc[model, method] for model in ordered_models]
        ax.bar(x + offset, values, width=width, color=METHOD_COLORS[method], label=_label(method), alpha=0.92)
    ax.set_xticks(x)
    ax.set_xticklabels([model_labels[m] for m in ordered_models])
    ax.set_ylabel("平均均方根误差（RMSE）")
    ax.set_title("不同未知噪声模型下的代表算法对比")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3)
    save_fig(fig, out_path)


def plot_parameter_sensitivity(parameter_table: pd.DataFrame, out_path: Path, default_lam: float, default_eta: float) -> None:
    _paper_style()
    pivot = parameter_table.pivot(index="median_weight", columns="amr_lambda", values="rmse")
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    im = ax.imshow(pivot.values, origin="lower", aspect="auto", cmap="viridis_r")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"{v:g}" for v in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([f"{v:g}" for v in pivot.index])
    ax.set_xlabel("AMR-GTV 图正则强度")
    ax.set_ylabel("自适应中值先验强度")
    ax.set_title("AMR-GTV 关键参数敏感性")
    if default_lam in list(pivot.columns) and default_eta in list(pivot.index):
        x = list(pivot.columns).index(default_lam)
        y = list(pivot.index).index(default_eta)
        ax.scatter([x], [y], s=90, marker="*", color="red", edgecolor="white", linewidth=0.8)
    fig.colorbar(im, ax=ax, label="RMSE")
    save_fig(fig, out_path)


def plot_ablation(ablation_table: pd.DataFrame, out_path: Path) -> None:
    _paper_style()
    df = ablation_table.sort_values("rmse_mean", ascending=False)
    fig, ax = plt.subplots(figsize=(7.2, 3.7))
    colors = ["#d62728" if v == "amr_gtv" else "#777777" for v in df["variant"]]
    ax.barh(df["variant_label"], df["rmse_mean"], color=colors, alpha=0.92)
    ax.set_xlabel("平均均方根误差（RMSE）")
    ax.set_title("AMR-GTV 组件消融实验")
    ax.grid(True, axis="x", alpha=0.28)
    save_fig(fig, out_path)


def plot_runtime_vs_points(main_summary: pd.DataFrame, out_path: Path) -> None:
    _paper_style()
    runtime = (
        main_summary.groupby(["n_points", "method", "method_label"], as_index=False)
        .agg(runtime_mean=("runtime_sec", "mean"))
        .sort_values(["n_points", "method"])
    )
    fig, ax = plt.subplots(figsize=(6.8, 3.7))
    for method in METHOD_ORDER:
        df = runtime[runtime["method"] == method].sort_values("n_points")
        ax.plot(
            df["n_points"],
            df["runtime_mean"],
            marker=METHOD_MARKERS[method],
            color=METHOD_COLORS[method],
            linewidth=2.2 if method == "amr_gtv" else 1.4,
            label=_label(method),
        )
    ax.set_xlabel("稀疏观测点数量")
    ax.set_ylabel("平均运行时间（s）")
    ax.set_title("不同观测规模下的运行时间对比")
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=False, ncol=2)
    save_fig(fig, out_path)


def plot_casewise_reduction(main_summary: pd.DataFrame, out_path: Path) -> None:
    _paper_style()
    keys = ["seed", "timestep", "sampling_rate", "noise_ratio"]
    amr = main_summary[main_summary["method"] == "amr_gtv"][keys + ["rmse"]].rename(columns={"rmse": "rmse_amr"})
    reductions = []
    labels = []
    colors = []
    for method in ["knn_mean", "median", "graph_laplacian"]:
        base = main_summary[main_summary["method"] == method][keys + ["rmse"]].rename(columns={"rmse": "rmse_base"})
        merged = base.merge(amr, on=keys)
        reductions.append((merged["rmse_base"] - merged["rmse_amr"]) / merged["rmse_base"] * 100.0)
        labels.append(_label(method))
        colors.append(METHOD_COLORS[method])

    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    parts = ax.violinplot(reductions, showmedians=True, showextrema=False)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_alpha(0.28)
        body.set_edgecolor("none")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("AMR-GTV 相对 RMSE 降低率（%）")
    ax.set_title("逐实验条件下本文方法相对代表基线的提升分布")
    ax.grid(True, axis="y", alpha=0.28)
    save_fig(fig, out_path)


def make_figures(data, main_summary: pd.DataFrame, tables: dict[str, pd.DataFrame], representative: dict, fig_dir: Path, args: argparse.Namespace) -> None:
    obs = representative["obs"]
    graph = build_knn_graph(obs.coords, k=args.k)
    plot_dataset_mask_sampling(data, obs, fig_dir / "fig_2_1_dataset_mask_sampling.png")
    plot_noise_model(obs, representative["y_noisy"], representative["noise"], fig_dir / "fig_2_2_noise_model.png")
    plot_noise_model_comparison(tables["noise_model"], fig_dir / "fig_2_3_noise_model_comparison.png")
    plot_knn_graph(obs, graph.neighbors, fig_dir / "fig_2_4_knn_graph.png")
    plot_focused_ranking(tables["ranking"], fig_dir / "fig_2_5_method_ranking.png")
    plot_gap_bar(tables["gap"], fig_dir / "fig_2_6_amr_gap_bar.png")
    plot_error_distribution(representative, fig_dir / "fig_2_7_error_distribution.png")
    plot_spatial_error_maps(representative, fig_dir / "fig_2_8_spatial_error_maps.png")
    plot_noise_ratio(tables["by_noise"], fig_dir / "fig_2_9_noise_ratio_rmse.png")
    plot_outlier_sweep(tables["outlier"], fig_dir / "fig_2_10_outlier_fraction_rmse.png")
    plot_sampling_rate(tables["by_sampling"], fig_dir / "fig_2_11_sampling_gain.png")
    plot_gain_heatmap(main_summary, fig_dir / "fig_2_12_amr_gain_heatmap.png")
    default_lam, default_eta = default_amr_reference(args)
    plot_parameter_sensitivity(tables["parameter"], fig_dir / "fig_2_13_parameter_sensitivity.png", default_lam, default_eta)
    plot_casewise_reduction(main_summary, fig_dir / "fig_2_14_casewise_reduction.png")
    plot_runtime_vs_points(main_summary, fig_dir / "fig_2_15_runtime_vs_points.png")
    plot_ablation(tables["ablation"], fig_dir / "fig_2_16_ablation.png")


def main() -> None:
    args = parse_args()
    out_root = ensure_dir(args.out)
    metrics_dir = ensure_dir(out_root / "metrics")
    fig_dir = ensure_dir(out_root / "figures")
    data = load_flow_npz(resolve_data_path(args.data))

    main_summary, representative = run_main_grid(args, data)
    outlier_summary = run_outlier_sweep(args, data)
    noise_model_summary = run_noise_model_sweep(args, data)
    parameter_summary = run_parameter_sensitivity(args, data)
    ablation_summary = run_ablation(args, data)
    tables = write_tables(main_summary, outlier_summary, noise_model_summary, parameter_summary, ablation_summary, metrics_dir)
    make_figures(data, main_summary, tables, representative, fig_dir, args)

    print("\nFocused stress experiments finished.")
    print(f"Metrics: {metrics_dir}")
    print(f"Figures: {fig_dir}")


if __name__ == "__main__":
    main()
