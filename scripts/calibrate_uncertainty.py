"""Calibrate UniDepth uncertainty → reliability weights from a static clip.

Input:
  --depth-stack   : .npy of shape (T,H,W) depth over a (mostly) static scene.
  --uncert-stack  : .npy of shape (T,H,W) corresponding UniDepth confidence/uncertainty.

Output:
  - Prints a recommended k for weight = exp(-k * uncert_norm), with uncert_norm = uncert / median_uncert.
  - Prints stats for variance vs uncertainty bins.
  - Optionally saves a CSV of the curve (bin centers and mean variance).

Usage:
python scripts/calibrate_uncertainty.py \
  --depth-stack static_depth.npy \
  --uncert-stack static_uncert.npy \
  --bins 20 \
  --curve-csv /tmp/uncert_curve.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--depth-stack", type=Path, required=True, help="(T,H,W) depth stack .npy from a static clip.")
    p.add_argument("--uncert-stack", type=Path, required=True, help="(T,H,W) uncertainty/confidence stack .npy matching the depth.")
    p.add_argument("--bins", type=int, default=20, help="Number of bins for the uncert→variance curve.")
    p.add_argument("--curve-csv", type=Path, help="Optional path to save bin centers and mean variance as CSV.")
    p.add_argument("--eps", type=float, default=1e-6, help="Epsilon to avoid divide-by-zero.")
    p.add_argument(
        "--chunk-frames",
        type=int,
        default=16,
        help="Frames per chunk when streaming variance/mean (progress-friendly, low RAM).",
    )
    return p.parse_args()


def load_stack(path: Path, name: str) -> np.ndarray:
    arr = np.load(path, mmap_mode="r")
    if arr.ndim != 3:
        raise ValueError(f"{name} stack must be (T,H,W), got {arr.shape}")
    return arr  # keep memmap to avoid pulling full stack into RAM


def compute_variance_stream(depth: np.ndarray, chunk: int) -> np.ndarray:
    """Streamed variance over time axis with chunked reading to show progress and avoid RAM spikes."""

    t, h, w = depth.shape
    sum_ = np.zeros((h, w), dtype=np.float64)
    sumsq = np.zeros_like(sum_)
    for start in tqdm(
        range(0, t, chunk),
        desc="Variance",
        leave=False,
        dynamic_ncols=True,
        mininterval=0.2,
    ):
        end = min(t, start + chunk)
        block = np.asarray(depth[start:end], dtype=np.float32)
        sum_ += block.sum(axis=0, dtype=np.float64)
        sumsq += np.square(block, dtype=np.float64).sum(axis=0)
    count = float(t)
    mean = sum_ / count
    var = (sumsq / count) - mean * mean
    return var.astype(np.float32)


def compute_mean_stream(arr: np.ndarray, chunk: int) -> np.ndarray:
    t, h, w = arr.shape
    sum_ = np.zeros((h, w), dtype=np.float64)
    for start in tqdm(
        range(0, t, chunk),
        desc="Mean",
        leave=False,
        dynamic_ncols=True,
        mininterval=0.2,
    ):
        end = min(t, start + chunk)
        block = np.asarray(arr[start:end], dtype=np.float32)
        sum_ += block.sum(axis=0, dtype=np.float64)
    mean = sum_ / float(t)
    return mean.astype(np.float32)


def bin_curve(uncert_mean: np.ndarray, var_depth: np.ndarray, bins: int):
    u = uncert_mean.flatten()
    v = var_depth.flatten()
    valid = np.isfinite(u) & np.isfinite(v)
    u = u[valid]
    v = v[valid]
    if len(u) == 0:
        raise ValueError("No valid pixels to bin.")
    u_min, u_max = u.min(), u.max()
    edges = np.linspace(u_min, u_max + 1e-8, bins + 1)
    bin_ids = np.digitize(u, edges) - 1
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    mean_var = np.zeros(bins, dtype=np.float32)
    counts = np.zeros(bins, dtype=np.int64)
    for b in range(bins):
        mask = bin_ids == b
        if mask.any():
            mean_var[b] = v[mask].mean()
            counts[b] = mask.sum()
        else:
            mean_var[b] = np.nan
    return bin_centers, mean_var, counts


def fit_k(uncert_mean: np.ndarray, var_depth: np.ndarray, eps: float) -> Tuple[float, float]:
    """Fit k in weight = exp(-k * (uncert/median_uncert)) to approximate target reliability."""
    u = uncert_mean.flatten()
    v = var_depth.flatten()
    valid = np.isfinite(u) & np.isfinite(v)
    u = u[valid]
    v = v[valid]
    if len(u) == 0:
        raise ValueError("No valid pixels to fit.")
    u_med = np.median(u) + eps
    v_med = np.median(v) + eps
    u_norm = u / u_med
    v_norm = v / v_med
    target = 1.0 / (1.0 + v_norm)  # higher variance -> lower target weight

    k_grid = np.logspace(-3, 2, 60)
    best_k, best_mse = None, float("inf")
    for k in k_grid:
        w = np.exp(-k * u_norm)
        mse = float(((w - target) ** 2).mean())
        if mse < best_mse:
            best_mse = mse
            best_k = k
    return best_k, best_mse


def main():
    args = parse_args()
    depth = load_stack(args.depth_stack, "depth")
    uncert = load_stack(args.uncert_stack, "uncertainty")

    if depth.shape != uncert.shape:
        raise ValueError(f"Depth {depth.shape} and uncertainty {uncert.shape} shapes differ.")

    # streamed progress-friendly stats
    var_depth = compute_variance_stream(depth, chunk=args.chunk_frames)
    uncert_mean = compute_mean_stream(uncert, chunk=args.chunk_frames)

    bin_centers, mean_var, counts = bin_curve(uncert_mean, var_depth, args.bins)
    best_k, best_mse = fit_k(uncert_mean, var_depth, args.eps)

    print(f"Depth stack: {depth.shape}, dtype={depth.dtype}")
    print(f"Uncertainty stack: {uncert.shape}, dtype={uncert.dtype}")
    print(f"Variance stats: min={var_depth.min():.6f}, max={var_depth.max():.6f}, median={np.median(var_depth):.6f}")
    print(f"Uncertainty stats: min={uncert_mean.min():.6f}, max={uncert_mean.max():.6f}, median={np.median(uncert_mean):.6f}")
    print(f"Recommended k for weight = exp(-k * uncert/median_uncert): k={best_k:.4g}, mse={best_mse:.4g}")

    if args.curve_csv:
        args.curve_csv.parent.mkdir(parents=True, exist_ok=True)
        data = np.stack([bin_centers, mean_var, counts], axis=1)
        header = "bin_center,mean_variance,count"
        np.savetxt(args.curve_csv, data, delimiter=",", header=header, comments="")
        print(f"Saved uncertainty→variance curve to {args.curve_csv}")


if __name__ == "__main__":
    main()
