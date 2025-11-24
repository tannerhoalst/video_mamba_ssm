"""Baseline temporal smoothing for depth sequences (EMA and temporal bilateral) + flicker metrics.

Usage:
python scripts/baseline_temporal_filters.py \
  --input-stack /path/to/depth_stack.npy \
  --output-dir /path/to/output_dir \
  --save-ema --save-bilateral \
  --ema-alpha 0.8 \
  --bilateral-sigma-rel 0.05 \
  --static-thresh-rel 0.02

Inputs:
- input-stack: .npy array shape (T, H, W), float16/float32.

Outputs (optional):
- ema.npy, bilateral.npy in output-dir (float32).

Metrics:
- Global mean absolute diff between consecutive frames (MAD).
- Static-region MAD using a simple threshold mask on frame-to-frame depth deltas.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-stack", type=Path, required=True, help="Path to depth stack .npy (T,H,W).")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory to save smoothed stacks.")
    parser.add_argument("--save-ema", action="store_true", help="Save EMA-smoothed depth stack.")
    parser.add_argument("--save-bilateral", action="store_true", help="Save temporal bilateral-smoothed depth stack.")
    parser.add_argument("--ema-alpha", type=float, default=0.8, help="EMA smoothing factor (0-1, higher=more smoothing).")
    parser.add_argument(
        "--bilateral-sigma-rel",
        type=float,
        default=0.05,
        help="Relative sigma for temporal bilateral (sigma = bilateral_sigma_rel * median_depth).",
    )
    parser.add_argument(
        "--static-thresh-rel",
        type=float,
        default=0.02,
        help="Static mask threshold as a fraction of median depth (|Δ| < thresh is static).",
    )
    parser.add_argument(
        "--output-dtype",
        choices=["float16", "float32", "float64"],
        default="float32",
        help="Datatype for saved outputs.",
    )
    parser.add_argument(
        "--median-sample-frames",
        type=int,
        default=32,
        help="Frames to sample for median estimate (keeps memory low).",
    )
    parser.add_argument(
        "--median-sample-pixels",
        type=int,
        default=1_000_000,
        help="Max pixels used for median estimate across sampled frames.",
    )
    return parser.parse_args()


def load_stack(path: Path) -> np.ndarray:
    arr = np.load(path, mmap_mode="r")
    if arr.ndim != 3:
        raise ValueError(f"Expected stack shape (T,H,W), got {arr.shape}")
    return arr


def estimate_median(stack: np.ndarray, sample_frames: int, sample_pixels: int) -> float:
    """Approximate median without loading full stack into RAM.

    Samples up to `sample_frames` evenly spaced frames and up to `sample_pixels`
    total pixels to keep memory bounded.
    """

    t, h, w = stack.shape
    frames = np.linspace(0, t - 1, min(sample_frames, t), dtype=int)
    max_per_frame = max(1, sample_pixels // len(frames))
    rng = np.random.default_rng(0)
    samples = []
    for idx in frames:
        flat = np.asarray(stack[idx]).ravel()  # small one-frame view
        if flat.size <= max_per_frame:
            samples.append(flat.astype(np.float32))
        else:
            choice = rng.choice(flat.size, size=max_per_frame, replace=False)
            samples.append(flat[choice].astype(np.float32))
    gathered = np.concatenate(samples)
    return float(np.median(gathered))


def stream_filters(
    stack: np.ndarray,
    alpha: float,
    sigma: float,
    static_thresh: float,
    save_ema: bool,
    save_bilateral: bool,
    output_dir: Path | None,
    output_dtype: np.dtype,
) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    """Stream EMA + temporal bilateral without holding full arrays in RAM.

    Optionally writes outputs to disk using memmaps when save_* flags are set.
    Returns flicker metrics for (raw, ema, bilat).
    """

    t, h, w = stack.shape
    eps = 1e-6

    # Allocate disk-backed outputs only if requested.
    ema_out = bilat_out = None
    if output_dir and save_ema:
        output_dir.mkdir(parents=True, exist_ok=True)
        ema_out = np.lib.format.open_memmap(output_dir / "ema.npy", mode="w+", dtype=output_dtype, shape=(t, h, w))
    if output_dir and save_bilateral:
        output_dir.mkdir(parents=True, exist_ok=True)
        bilat_out = np.lib.format.open_memmap(output_dir / "bilateral.npy", mode="w+", dtype=output_dtype, shape=(t, h, w))

    # Running metrics accumulators
    raw_sum = raw_static_sum = 0.0
    raw_count = raw_static_count = 0
    ema_sum = ema_static_sum = 0.0
    ema_count = ema_static_count = 0
    bilat_sum = bilat_static_sum = 0.0
    bilat_count = bilat_static_count = 0

    # Initialize first frame
    prev_raw = np.asarray(stack[0], dtype=np.float32)
    prev_ema = prev_raw.copy()
    prev_bilat = prev_raw.copy()
    if ema_out is not None:
        ema_out[0] = prev_ema.astype(output_dtype)
    if bilat_out is not None:
        bilat_out[0] = prev_bilat.astype(output_dtype)

    for idx in tqdm(
        range(1, t),
        desc="Temporal smoothing",
        leave=False,
        dynamic_ncols=True,
        mininterval=0.1,
    ):
        curr = np.asarray(stack[idx], dtype=np.float32)

        # Raw metrics
        diff_raw = np.abs(curr - prev_raw)
        raw_sum += float(diff_raw.sum())
        raw_count += diff_raw.size
        raw_mask = diff_raw < static_thresh
        raw_static_sum += float((diff_raw * raw_mask).sum())
        raw_static_count += int(raw_mask.sum())

        # EMA
        ema_curr = alpha * prev_ema + (1.0 - alpha) * curr
        diff_ema = np.abs(ema_curr - prev_ema)
        ema_sum += float(diff_ema.sum())
        ema_count += diff_ema.size
        ema_mask = diff_ema < static_thresh
        ema_static_sum += float((diff_ema * ema_mask).sum())
        ema_static_count += int(ema_mask.sum())
        if ema_out is not None:
            ema_out[idx] = ema_curr.astype(output_dtype)

        # Temporal bilateral
        weight_prev = np.exp(-np.abs(curr - prev_bilat) / (sigma + eps))
        bilat_curr = (curr + weight_prev * prev_bilat) / (1.0 + weight_prev)
        diff_bilat = np.abs(bilat_curr - prev_bilat)
        bilat_sum += float(diff_bilat.sum())
        bilat_count += diff_bilat.size
        bilat_mask = diff_bilat < static_thresh
        bilat_static_sum += float((diff_bilat * bilat_mask).sum())
        bilat_static_count += int(bilat_mask.sum())
        if bilat_out is not None:
            bilat_out[idx] = bilat_curr.astype(output_dtype)

        prev_raw = curr
        prev_ema = ema_curr
        prev_bilat = bilat_curr

    def finalize(total_sum: float, count: int, static_sum: float, static_count: int) -> Tuple[float, float]:
        global_mad = total_sum / count if count else float("nan")
        static_mad = static_sum / static_count if static_count else float("nan")
        return global_mad, static_mad

    raw_metrics = finalize(raw_sum, raw_count, raw_static_sum, raw_static_count)
    ema_metrics = finalize(ema_sum, ema_count, ema_static_sum, ema_static_count)
    bilat_metrics = finalize(bilat_sum, bilat_count, bilat_static_sum, bilat_static_count)
    return raw_metrics, ema_metrics, bilat_metrics


def main():
    args = parse_args()
    stack = load_stack(args.input_stack)
    median_depth = estimate_median(stack, args.median_sample_frames, args.median_sample_pixels)
    sigma = args.bilateral_sigma_rel * median_depth
    static_thresh = args.static_thresh_rel * median_depth

    print(
        f"Loaded stack {stack.shape}, dtype={stack.dtype}, approx_median_depth={median_depth:.4f} "
        f"(sampled {args.median_sample_frames} frames / {args.median_sample_pixels} pixels)")
    print(f"EMA alpha={args.ema_alpha}")
    print(f"Bilateral sigma={sigma:.6f} (rel={args.bilateral_sigma_rel})")
    print(f"Static threshold={static_thresh:.6f} (rel={args.static_thresh_rel})")

    raw_metrics, ema_metrics, bilat_metrics = stream_filters(
        stack,
        alpha=args.ema_alpha,
        sigma=sigma,
        static_thresh=static_thresh,
        save_ema=args.save_ema,
        save_bilateral=args.save_bilateral,
        output_dir=args.output_dir,
        output_dtype=np.dtype(args.output_dtype),
    )

    print("\nFlicker (mean abs diff frame-to-frame):")
    print(f"  Raw   : global={raw_metrics[0]:.6f}  static={raw_metrics[1]:.6f}")
    print(f"  EMA   : global={ema_metrics[0]:.6f}  static={ema_metrics[1]:.6f}")
    print(f"  Bilat : global={bilat_metrics[0]:.6f}  static={bilat_metrics[1]:.6f}")

    if args.output_dir:
        if args.save_ema:
            print(f"Saved EMA to {Path(args.output_dir) / 'ema.npy'}")
        if args.save_bilateral:
            print(f"Saved bilateral to {Path(args.output_dir) / 'bilateral.npy'}")


if __name__ == "__main__":
    main()
