"""
Geometric transforms for video depth pipeline.

Responsibilities:
- Resize + pad while keeping metric depth unchanged and updating intrinsics.
- Produce aligned low-res and full-res tensors (RGB, depth, uncertainty).
- Emit explicit mapping tensor M (scale factors + padding offsets) for low→full upsampling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


@dataclass
class ResizePadResult:
    rgb: torch.Tensor           # [3, Ht, Wt]
    depth: torch.Tensor         # [1, Ht, Wt]
    uncertainty: torch.Tensor   # [1, Ht, Wt]
    intrinsics: Dict[str, float]
    scale: float
    pad: Tuple[int, int, int, int]  # (pad_left, pad_right, pad_top, pad_bottom)


def resize_and_pad(
    rgb: torch.Tensor,
    depth: torch.Tensor,
    uncertainty: torch.Tensor,
    intrinsics: Dict[str, float],
    target_hw: Tuple[int, int],
) -> ResizePadResult:
    """
    Resize with aspect lock to fit inside target_hw, then center-pad to exact size.
    Depth values remain in meters; only intrinsics are scaled/shifted.
    """
    _, h, w = rgb.shape
    target_h, target_w = target_hw

    scale = min(target_h / h, target_w / w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    # Resize
    def _interp(x, mode):
        return F.interpolate(x.unsqueeze(0), size=(new_h, new_w), mode=mode, align_corners=False).squeeze(0)

    rgb_r = _interp(rgb, mode="bilinear")
    depth_r = _interp(depth, mode="bilinear")
    uncert_r = _interp(uncertainty, mode="bilinear")

    # Pad to target
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left
    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    pad = (pad_left, pad_right, pad_top, pad_bottom)

    rgb_p = F.pad(rgb_r, pad, mode="replicate")
    depth_p = F.pad(depth_r, pad, mode="replicate")
    uncert_p = F.pad(uncert_r, pad, mode="replicate")

    # Intrinsics update
    fx = intrinsics["fx"] * scale
    fy = intrinsics["fy"] * scale
    cx = intrinsics["cx"] * scale + pad_left
    cy = intrinsics["cy"] * scale + pad_top
    intr_out = {**intrinsics, "fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy)}

    return ResizePadResult(
        rgb=rgb_p,
        depth=depth_p,
        uncertainty=uncert_p,
        intrinsics=intr_out,
        scale=scale,
        pad=pad,
    )


@dataclass
class MultiScaleBatch:
    rgb_full: torch.Tensor        # [3, Hf, Wf]
    depth_full: torch.Tensor      # [1, Hf, Wf]
    uncert_full: torch.Tensor     # [1, Hf, Wf]
    intrinsics_full: Dict[str, float]
    rgb_low: torch.Tensor         # [3, Hl, Wl]
    depth_low: torch.Tensor       # [1, Hl, Wl]
    uncert_low: torch.Tensor      # [1, Hl, Wl]
    intrinsics_low: Dict[str, float]
    mapping_tensor: torch.Tensor  # [4, Hl, Wl] -> (scale_y, scale_x, pad_top, pad_left)
    mapping_meta: Dict[str, float]


def make_multiscale(
    rgb: torch.Tensor,
    depth: torch.Tensor,
    uncertainty: torch.Tensor,
    intrinsics: Dict[str, float],
    target_full: Tuple[int, int],
    target_low: Tuple[int, int],
) -> MultiScaleBatch:
    """
    Produce full-res (target_full) and aligned low-res (target_low) tensors plus mapping.
    """
    full = resize_and_pad(rgb, depth, uncertainty, intrinsics, target_full)

    # Low-res derived from padded full for consistent origin
    scale_low_y = target_low[0] / target_full[0]
    scale_low_x = target_low[1] / target_full[1]

    def _down(x, mode):
        return F.interpolate(
            x.unsqueeze(0),
            size=target_low,
            mode=mode,
            align_corners=False,
        ).squeeze(0)

    rgb_low = _down(full.rgb, mode="bilinear")
    depth_low = _down(full.depth, mode="bilinear")
    uncert_low = _down(full.uncertainty, mode="bilinear")

    # Intrinsics for low-res
    fx_l = full.intrinsics["fx"] * scale_low_x
    fy_l = full.intrinsics["fy"] * scale_low_y
    cx_l = full.intrinsics["cx"] * scale_low_x
    cy_l = full.intrinsics["cy"] * scale_low_y
    intr_low = {**full.intrinsics, "fx": float(fx_l), "fy": float(fy_l), "cx": float(cx_l), "cy": float(cy_l)}

    # Mapping tensor M: scale factors + padding to map low→full grid
    sy = 1.0 / scale_low_y  # low → full
    sx = 1.0 / scale_low_x
    Hl, Wl = target_low
    M = torch.stack(
        [
            torch.full((Hl, Wl), sy, dtype=torch.float32),
            torch.full((Hl, Wl), sx, dtype=torch.float32),
            torch.full((Hl, Wl), full.pad[2], dtype=torch.float32),  # pad_top
            torch.full((Hl, Wl), full.pad[0], dtype=torch.float32),  # pad_left
        ],
        dim=0,
    )

    mapping_meta = {
        "scale_y": sy,
        "scale_x": sx,
        "pad_top": float(full.pad[2]),
        "pad_left": float(full.pad[0]),
        "H_full": target_full[0],
        "W_full": target_full[1],
        "H_low": Hl,
        "W_low": Wl,
    }

    return MultiScaleBatch(
        rgb_full=full.rgb,
        depth_full=full.depth,
        uncert_full=full.uncertainty,
        intrinsics_full=full.intrinsics,
        rgb_low=rgb_low,
        depth_low=depth_low,
        uncert_low=uncert_low,
        intrinsics_low=intr_low,
        mapping_tensor=M,
        mapping_meta=mapping_meta,
    )


def check_no_nans(*tensors: torch.Tensor) -> None:
    for t in tensors:
        if torch.isnan(t).any() or torch.isinf(t).any():
            raise ValueError("Found NaN/Inf in tensor during transform pipeline.")
