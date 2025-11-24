"""Benchmark UniDepth V2 inference in a single process (no reload between runs).

This captures model load once, then runs multiple forward passes on the same
image. Timings therefore reflect preprocessing + forward pass, not checkpoint
loading. Use CUDA for realistic speed if available.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from unidepth.models import UniDepthV2

# cache models by (model_id, device_str) to avoid reloading weights
_MODEL_CACHE: dict[tuple[str, str], UniDepthV2] = {}


def get_unidepth_model(model_id: str, device: torch.device) -> UniDepthV2:
    key = (model_id, str(device))
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = UniDepthV2.from_pretrained(model_id).to(device).eval()
    return _MODEL_CACHE[key]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=60, help="Number of timed runs")
    parser.add_argument(
        "--warmup", type=int, default=5, help="Warm-up runs (not timed)"
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=Path("/home/thoalst/Pictures/Screenshots/savannah.png"),
        help="Path to input RGB image",
    )
    parser.add_argument(
        "--model-id",
        default="lpiccinelli/unidepth-v2-vits14",
        help="Model ID for UniDepthV2.from_pretrained",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device string (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--save-depth",
        type=Path,
        default=None,
        help="Optional path to save the depth from the *last* timed run",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-run timings (default: only summary)",
    )
    return parser.parse_args()


def load_rgb_tensor(image_path: Path) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    rgb = torch.from_numpy(np.array(image))
    return rgb.permute(2, 0, 1).contiguous()  # C, H, W


def maybe_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    args = parse_args()

    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # Load model once (cached)
    load_start = time.perf_counter()
    model = get_unidepth_model(args.model_id, device)
    load_time = time.perf_counter() - load_start
    print(f"Model loaded (or retrieved from cache) in {load_time:.2f} s")

    # Preload input tensor once (already on target device).
    rgb = load_rgb_tensor(args.image).to(device, non_blocking=True)

    durations = []

    def run_infer() -> torch.Tensor:
        with torch.no_grad():
            maybe_sync(device)
            t0 = time.perf_counter()
            preds = model.infer(rgb, normalize=True)
            maybe_sync(device)
            return preds["depth"][0], time.perf_counter() - t0

    # Warm-up
    for i in range(args.warmup):
        print("Input tensor shape:", tuple(rgb.shape))
        depth, dt = run_infer()
        print(f"Warm-up {i+1}/{args.warmup}: {dt:.3f} s")

    # Timed runs
    last_depth: Optional[torch.Tensor] = None
    for i in range(args.runs):
        depth, dt = run_infer()
        durations.append(dt)
        last_depth = depth
        if args.verbose:
            print(f"Run {i+1}/{args.runs}: {dt:.3f} s")

    mean = statistics.mean(durations)
    median = statistics.median(durations)
    p95 = statistics.quantiles(durations, n=100)[94] if len(durations) >= 2 else mean
    total = sum(durations)

    print("\nTimed runs summary (no reload between runs):")
    print(f"- runs:   {args.runs}")
    print(f"- mean:   {mean:.5f} s")
    print(f"- median: {median:.5f} s")
    print(f"- p95:    {p95:.5f} s")
    print(f"- min:    {min(durations):.5f} s")
    print(f"- max:    {max(durations):.5f} s")
    print(f"- total:  {total:.5f} s")

    if args.save_depth and last_depth is not None:
        args.save_depth.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.save_depth, last_depth.cpu().numpy())
        print(f"Saved depth from last run to {args.save_depth}")


if __name__ == "__main__":
    main()
