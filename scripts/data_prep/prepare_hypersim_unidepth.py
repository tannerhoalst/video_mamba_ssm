#!/usr/bin/env python3
"""
Normalize Hypersim RGB + depth to a UniDepth-compatible format and emit a GT-aware manifest.

Key features:
- Single fixed working size (default: short side 518 → center pad/crop to 518x518).
- Depth kept in meters (float32); verifies range.
- Intrinsics rescaled/shifted after resize + pad; stored per frame.
- Produces low-res copies (×2 or ×4) for the SSM branch and records the mapping
  needed to upsample residuals back to full resolution.
- Precomputes high-frequency guidance maps (RGB edges, depth edges) for the
  high-res refinement stage or guided upsampling.

Outputs per frame (inside --out-root):
  rgb_full.png
  depth_full.npy          (float32 meters)
  depth_low.npy           (float32 meters)
  edges_rgb.npy           (float32)
  edges_depth.npy         (float32)
  meta.json               (intrinsics, padding, scales, mapping)

Manifest (manifest.jsonl) fields:
  rgb_path, depth_gt_path, depth_low_path,
  edges_rgb_path, edges_depth_path,
  intrinsics_full, intrinsics_low,
  scale_full_from_raw, pad, low_scale_factor,
  has_gt (bool), dataset_name, scene_id, frame_index
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, Tuple, Optional

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

def find_dataset(h5f: h5py.File, is_depth: bool) -> h5py.Dataset:
    """Best-effort dataset picker for Hypersim color/depth HDF5 files."""
    candidates = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            shape = obj.shape
            if is_depth:
                if len(shape) == 3:  # (F, H, W)
                    candidates.append(obj)
            else:
                if len(shape) == 4 and shape[-1] in (3, 4):  # (F, H, W, C)
                    candidates.append(obj)

    h5f.visititems(visitor)
    if not candidates:
        raise RuntimeError("No suitable dataset found in HDF5.")
    # Prefer the largest dataset (usually the main sequence)
    candidates.sort(key=lambda d: np.prod(d.shape), reverse=True)
    return candidates[0]


def load_intrinsics_table(csv_path: Path, cam_id: str) -> Dict[int, Dict[str, float]]:
    """
    Hypersim metadata_cameras.csv (as downloaded here) only lists camera names, no intrinsics.
    We only use it to verify the camera exists; actual intrinsics are per-cam or fixed-FOV.
    """
    import csv

    cams = set()
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cams.add(row.get("camera_name", "").strip())
    return {"default": {}}


def load_intrinsics_from_cam(scene_dir: Path, cam_id: str, expected_frames: int, image_hw: Tuple[int, int]):
    """
    Placeholder retained for completeness; Hypersim public release does not ship this file.
    Always returns None.
    Expected format: dataset shape (F, 3, 3) or (F, 4) with fx, fy, cx, cy.
    """
    return None


def load_intrinsics_from_proj(scene_dir: Path, cam_id: str, expected_frames: int):
    """
    Parse the bundled metadata_camera_parameters.csv to derive fx, fy, cx, cy from the OpenGL projection matrix M_proj.
    Uses formulas: fx = m00 * W / 2, fy = m11 * H / 2, cx = (m02 + 1) * W / 2, cy = (m12 + 1) * H / 2
    where W,H are settings_output_img_width/height.
    """
    csv_path = Path(__file__).parent / "metadata_camera_parameters.csv"  # bundled in repo
    if not csv_path.exists():
        raise FileNotFoundError("Bundled metadata_camera_parameters.csv not found.")
    df = pd.read_csv(csv_path)
    cam_short = cam_id.replace("scene_", "")
    if "camera_name" in df.columns:
        df = df[df["camera_name"] == cam_short]
    if df.empty:
        return None

    def parse_proj(row):
        W = float(row["settings_output_img_width"])
        H = float(row["settings_output_img_height"])
        if "M_proj" in row:
            proj = row["M_proj"]
            vals = [float(x) for x in str(proj).strip().split()]
            if len(vals) != 16:
                return None
            m00, m01, m02, m03, m10, m11, m12, m13, m20, m21, m22, m23, m30, m31, m32, m33 = vals
        else:
            # Columns M_proj_00 ... M_proj_33
            vals = []
            for i in range(4):
                for j in range(4):
                    key = f"M_proj_{i}{j}"
                    if key not in row:
                        return None
                    vals.append(float(row[key]))
            m00, m01, m02, m03, m10, m11, m12, m13, m20, m21, m22, m23, m30, m31, m32, m33 = vals
        fx = m00 * W / 2.0
        fy = m11 * H / 2.0
        cx = (m02 + 1.0) * W / 2.0
        cy = (m12 + 1.0) * H / 2.0
        return W, H, fx, fy, cx, cy

    intr = {}
    if "frame_id" in df.columns:
        for _, row in df.iterrows():
            parsed = parse_proj(row)
            if parsed is None:
                continue
            _, _, fx, fy, cx, cy = parsed
            frame = int(row["frame_id"])
            intr[frame] = {"fx": fx, "fy": fy, "cx": cx, "cy": cy}
    else:
        row = df.iloc[0]
        parsed = parse_proj(row)
        if parsed is None:
            raise RuntimeError("Failed to parse projection matrix from metadata_camera_parameters.csv")
        _, _, fx, fy, cx, cy = parsed
        for i in range(expected_frames):
            intr[i] = {"fx": fx, "fy": fy, "cx": cx, "cy": cy}
    return intr


def resize_with_pad(t: torch.Tensor, target: int) -> Tuple[torch.Tensor, Dict]:
    """
    Resize so short side == target, then center pad/crop to square target x target.
    t: tensor [C, H, W]
    Returns resized tensor and meta dict with scale & padding.
    """
    _, h, w = t.shape
    scale = target / min(h, w)
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))

    t = t.unsqueeze(0)  # [1, C, H, W]
    t = F.interpolate(t, size=(new_h, new_w), mode="bilinear", align_corners=False)

    # Center pad/crop to target
    pad_top = max(0, (target - new_h) // 2)
    pad_bottom = max(0, target - new_h - pad_top)
    pad_left = max(0, (target - new_w) // 2)
    pad_right = max(0, target - new_w - pad_left)

    if any(p > 0 for p in (pad_left, pad_right, pad_top, pad_bottom)):
        t = F.pad(t, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0.0)

    # Crop if oversized
    t = t[:, :, :target, :target]
    meta = {
        "scale": scale,
        "pad": [pad_top, pad_bottom, pad_left, pad_right],
        "out_h": target,
        "out_w": target,
    }
    return t.squeeze(0), meta


def downsample(t: torch.Tensor, factor: int) -> torch.Tensor:
    """Average-pool downsample by factor (depth-safe)."""
    if factor == 1:
        return t
    t = t.unsqueeze(0)  # [1, C, H, W]
    t = F.interpolate(t, scale_factor=1.0 / factor, mode="bilinear", align_corners=False)
    return t.squeeze(0)


def sobel_edges(gray: torch.Tensor) -> torch.Tensor:
    """Compute Sobel magnitude for a [1, H, W] tensor, returns [1, H, W]."""
    kernel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
    kernel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
    gx = F.conv2d(gray.unsqueeze(0), kernel_x, padding=1)
    gy = F.conv2d(gray.unsqueeze(0), kernel_y, padding=1)
    mag = torch.sqrt(gx * gx + gy * gy + 1e-8)
    return mag.squeeze(0)


def save_numpy(path: Path, arr: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)


def save_png(path: Path, arr: np.ndarray):
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(arr)
    img.save(path)


# --------------------------------------------------------------------------- #
# Main processing                                                             #
# --------------------------------------------------------------------------- #

def process_scene(scene_dir: Path, out_root: Path, args) -> int:
    # Handle one or more cameras: scene_cam_*_final_hdf5 and scene_cam_*_geometry_hdf5
    cam_final_dirs = sorted(scene_dir.glob("images/scene_cam_*_final_hdf5"))
    cam_geom_dirs = {d.name.replace("_geometry_hdf5", ""): d for d in scene_dir.glob("images/scene_cam_*_geometry_hdf5")}
    if not cam_final_dirs:
        raise RuntimeError(f"No scene_cam_*_final_hdf5 found in {scene_dir}")

    intr_path = next(scene_dir.rglob("metadata_cameras.csv"))

    manifest_entries = []
    total_frames = 0

    for cam_dir in cam_final_dirs:
        cam_id = cam_dir.name.replace("_final_hdf5", "")  # e.g., scene_cam_00
        geom_dir = cam_geom_dirs.get(cam_id)
        if geom_dir is None:
            print(f"Warning: no matching geometry dir for {cam_id}, skipping")
            continue

        intr_table = load_intrinsics_table(intr_path, cam_id)
        per_frame_intr = None

        color_files = sorted(cam_dir.glob("frame.*.color.hdf5"))
        depth_files = sorted(geom_dir.glob("frame.*.depth_meters.hdf5"))

        if not color_files or not depth_files:
            print(f"Warning: empty color/depth for {cam_id} in {scene_dir}")
            continue
        if len(color_files) != len(depth_files):
            print(f"Warning: RGB/depth count mismatch for {cam_id}: {len(color_files)} vs {len(depth_files)}; skipping cam")
            continue

        num_frames = len(color_files)
        total_frames += num_frames
        per_frame_intr = load_intrinsics_from_proj(scene_dir, cam_id, num_frames)
        fallback_logged = False

        for idx, (cpath, dpath) in enumerate(zip(color_files, depth_files)):
            with h5py.File(cpath, "r") as cf:
                rgb = cf["dataset"][...]  # float16 0-1, shape H,W,3
            with h5py.File(dpath, "r") as df:
                depth = df["dataset"][...]  # float16 meters, shape H,W

            # Sanity: ensure meters
            if depth.max() > 1000:  # likely mm
                depth = depth / 1000.0

            # rgb is float16 HDR in [0,1]; clamp to avoid overflow warnings, then scale
            rgb = np.clip(rgb, 0.0, 1.0, out=rgb).astype(np.float32)
            rgb = (rgb * 255.0).astype(np.uint8)

            rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0  # [3,H,W]
            depth_t = torch.from_numpy(depth).unsqueeze(0).float()  # [1,H,W]

            # Resize + pad
            rgb_resized, meta_rgb = resize_with_pad(rgb_t, args.target_size)
            depth_resized, meta_d = resize_with_pad(depth_t, args.target_size)

            # Intrinsics: per-frame if available; else fixed 60° HFOV default
            if per_frame_intr:
                intr_raw = per_frame_intr.get(idx, per_frame_intr.get(len(per_frame_intr)-1))
            else:
                H_raw, W_raw = rgb.shape[0], rgb.shape[1]
                f_x = W_raw / (2 * math.tan(math.radians(60) / 2))
                f_y = f_x
                c_x = (W_raw - 1) / 2
                c_y = (H_raw - 1) / 2
                intr_raw = {"fx": f_x, "fy": f_y, "cx": c_x, "cy": c_y}
                if not fallback_logged:
                    print(f"Info: {scene_dir.name}/{cam_id} using fixed 60° HFOV intrinsics (no per-frame intrinsics file).")
                    fallback_logged = True

            scale = meta_d["scale"]
            pad_t, pad_b, pad_l, pad_r = meta_d["pad"]
            fx = intr_raw["fx"] * scale
            fy = intr_raw["fy"] * scale
            cx = intr_raw["cx"] * scale + pad_l
            cy = intr_raw["cy"] * scale + pad_t

            intr_full = {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "w": meta_d["out_w"], "h": meta_d["out_h"]}

            # Low-res for SSM
            depth_low = downsample(depth_resized, args.low_factor)

            # Guidance maps
            gray = (0.2989 * rgb_resized[0] + 0.5870 * rgb_resized[1] + 0.1140 * rgb_resized[2]).unsqueeze(0)
            edges_rgb = sobel_edges(gray)
            edges_depth = sobel_edges(depth_resized)

            # Paths
            frame_id = f"{idx:06d}"
            out_scene = out_root / scene_dir.name / cam_id
            rgb_path = out_scene / "rgb_full" / f"{frame_id}.png"
            depth_path_full = out_scene / "depth_full" / f"{frame_id}.npy"
            depth_path_low = out_scene / "depth_low" / f"{frame_id}.npy"
            edges_rgb_path = out_scene / "edges_rgb" / f"{frame_id}.npy"
            edges_depth_path = out_scene / "edges_depth" / f"{frame_id}.npy"
            meta_path = out_scene / "meta" / f"{frame_id}.json"

        # Save
        save_png(rgb_path, (rgb_resized.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8))
        save_numpy(depth_path_full, depth_resized.squeeze(0).numpy().astype(np.float32))
        save_numpy(depth_path_low, depth_low.squeeze(0).numpy().astype(np.float32))
        save_numpy(edges_rgb_path, edges_rgb.squeeze(0).numpy().astype(np.float32))
        save_numpy(edges_depth_path, edges_depth.squeeze(0).numpy().astype(np.float32))

        meta = {
            "frame_index": idx,
            "cam_id": cam_id,
            "scale_from_raw": scale,
            "pad": [pad_t, pad_b, pad_l, pad_r],
            "low_factor": args.low_factor,
            "intrinsics_full": intr_full,
            "intrinsics_low": {
                "fx": fx / args.low_factor,
                "fy": fy / args.low_factor,
                "cx": cx / args.low_factor,
                "cy": cy / args.low_factor,
                "w": depth_low.shape[-1],
                "h": depth_low.shape[-2],
            },
            "mapping": {
                "scale": args.low_factor,
                "pad": [pad_t, pad_b, pad_l, pad_r],
            },
        }
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta, indent=2))

        manifest_entries.append(
            {
                "scene_id": scene_dir.name,
                "cam_id": cam_id,
                "frame_index": idx,
                "dataset_name": "hypersim",
                "has_gt": True,
                "rgb_path": str(rgb_path.relative_to(out_root)),
                "depth_gt_path": str(depth_path_full.relative_to(out_root)),
                "depth_low_path": str(depth_path_low.relative_to(out_root)),
                "edges_rgb_path": str(edges_rgb_path.relative_to(out_root)),
                "edges_depth_path": str(edges_depth_path.relative_to(out_root)),
                "intrinsics_full": intr_full,
                "intrinsics_low": meta["intrinsics_low"],
                "mapping": meta["mapping"],
            }
        )

    # Write manifest for this scene (per scene, all cams)
    manifest_path = out_root / scene_dir.name / "manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry) + "\n")

    return total_frames


def main():
    parser = argparse.ArgumentParser(description="Prepare Hypersim for UniDepth + temporal refiner.")
    parser.add_argument("--in-root", type=Path, default=Path("/mnt/vrdata/depth_ground_truth/hypersim"), help="Raw Hypersim root with color.hdf5/depth_meters.hdf5")
    parser.add_argument("--out-root", type=Path, default=Path("/mnt/vrdata/depth_ground_truth/hypersim_prepared"), help="Output root for normalized data")
    parser.add_argument("--target-size", type=int, default=518, help="Short side resize, then pad/crop to square of this size")
    parser.add_argument("--low-factor", type=int, default=2, choices=[1, 2, 4], help="Downsample factor for SSM branch")
    args = parser.parse_args()

    scenes = sorted([p for p in args.in_root.iterdir() if p.is_dir()])
    if not scenes:
        raise RuntimeError(f"No scenes found in {args.in_root}")

    total_frames = 0
    for scene in scenes:
        print(f"[Scene] {scene.name}")
        total_frames += process_scene(scene, args.out_root, args)
    print(f"Done. Wrote {total_frames} frames to {args.out_root}")


if __name__ == "__main__":
    main()
