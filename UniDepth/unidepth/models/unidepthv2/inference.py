#!/usr/bin/env python3
"""Simple CLI to run UniDepthV2 inference on a single RGB image."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from unidepth.models import UniDepthV2


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        required=True,
        type=Path,
        help="Path to the input RGB image.",
    )
    parser.add_argument(
        "--model-id",
        default="lpiccinelli/unidepth-v2-vits14",
        help="Hugging Face repo ID or local checkpoint for UniDepthV2.from_pretrained.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device string (default: cuda if available else cpu).",
    )
    parser.add_argument(
        "--save-depth",
        type=Path,
        default=None,
        help="Optional path to store the metric depth map as .npy.",
    )
    parser.add_argument(
        "--save-points",
        type=Path,
        default=None,
        help="Optional path to store the 3D point cloud (camera coordinates) as .npy.",
    )
    parser.add_argument(
        "--save-intrinsics",
        type=Path,
        default=None,
        help="Optional path to store the predicted intrinsics matrix as .npy.",
    )
    return parser.parse_args()


def _load_rgb_tensor(image_path: Path) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    rgb = torch.from_numpy(np.array(image))
    rgb = rgb.permute(2, 0, 1).contiguous()  # C, H, W
    return rgb


def _save_if_requested(tensor: torch.Tensor, target: Optional[Path]) -> None:
    if target is None:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    np.save(target, tensor.cpu().numpy())


def main() -> None:
    args = _parse_args()

    if not args.image.exists():
        raise FileNotFoundError(f"Input image not found: {args.image}")

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    model = UniDepthV2.from_pretrained(args.model_id)
    model = model.to(device).eval()

    rgb = _load_rgb_tensor(args.image)

    with torch.no_grad():
        predictions = model.infer(rgb, normalize=True)

    depth = predictions["depth"][0]
    points = predictions["points"][0]
    intrinsics = predictions["intrinsics"][0]

    print(f"Inference finished on {args.image}")
    print(f"Depth map shape: {tuple(depth.shape)} (meters)")
    print(f"Point cloud shape: {tuple(points.shape)} (camera coordinates)")
    print("Camera intrinsics:\n", intrinsics.cpu().numpy())

    _save_if_requested(depth, args.save_depth)
    _save_if_requested(points, args.save_points)
    _save_if_requested(intrinsics, args.save_intrinsics)


if __name__ == "__main__":
    main()
