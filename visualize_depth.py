"""Convert a UniDepth .npy depth map to a colored PNG (Inferno)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize


def load_depth(path: Path) -> np.ndarray:
    depth = np.load(path)
    depth = np.squeeze(depth)
    if depth.ndim != 2:
        raise ValueError(f"Expected a 2D depth array after squeeze, got shape {depth.shape}")
    return depth


def compute_norm(depth: np.ndarray, lower: float, upper: float) -> Normalize:
    finite = np.isfinite(depth)
    if not finite.any():
        raise ValueError("Depth array contains no finite values.")
    vmin, vmax = np.percentile(depth[finite], [lower, upper])
    if vmin == vmax:
        vmax = vmin + 1e-6
    return Normalize(vmin=vmin, vmax=vmax, clip=True)


def save_colormap(depth: np.ndarray, out_path: Path, norm: Normalize, cmap: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Some Matplotlib versions don't support `norm` in imsave; use vmin/vmax instead.
    plt.imsave(out_path, depth, cmap=cmap, vmin=norm.vmin, vmax=norm.vmax)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "npy_path",
        type=Path,
        help="Path to the depth .npy file (e.g., /mnt/vrdata/depth_maps/unidepth/savannah.npy)",
    )
    parser.add_argument(
        "--cmap",
        default="inferno",
        help="Matplotlib colormap name (default: inferno)",
    )
    parser.add_argument(
        "--percentiles",
        type=float,
        nargs=2,
        default=(1.0, 99.0),
        metavar=("LOW", "HIGH"),
        help="Percentile range for contrast stretching (default: 1 99)",
    )
    parser.add_argument(
        "--suffix",
        default="_inferno.png",
        help="Suffix for the output file name (default: _inferno.png)",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        help="Optional explicit output PNG path. If omitted, uses input name plus suffix in same directory.",
    )
    args = parser.parse_args()

    depth = load_depth(args.npy_path)
    norm = compute_norm(depth, *args.percentiles)

    if args.out:
        out_path = args.out
        if out_path.suffix.lower() != ".png":
            out_path = out_path.with_suffix(".png")
    else:
        out_path = args.npy_path.with_suffix("")  # strip .npy
        out_path = out_path.with_name(out_path.name + args.suffix)
    save_colormap(depth, out_path, norm, args.cmap)

    print(f"Saved colored depth to: {out_path}")


if __name__ == "__main__":
    main()
