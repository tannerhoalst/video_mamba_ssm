"""
Precompute teacher depth, uncertainty, SSM low-res copies, guidance maps, and optional flow/occlusion
for frames listed in the prepared Hypersim manifest.

Input:
  - Prepared dataset root (from prepare_hypersim_unidepth.py), which contains per-scene manifest.jsonl
    with rgb_full paths and intrinsics/mapping info.

Output (per scene/cam):
  depth_teacher_full/*.npy       float32 meters
  uncert_teacher_full/*.npy      float32
  depth_teacher_low/*.npy        float32
  uncert_teacher_low/*.npy       float32
  rgb_depth_grid/*.png           uint8 RGB downsampled to teacher depth grid (if needed)
  edges_rgb/*.npy, edges_depth/*.npy, gradmag/*.npy
  (optional) flow_low/*.npy, occ_low/*.npy
  Updated manifest.jsonl with new paths/metadata

Notes:
  - Uses bundled intrinsics CSV if per-entry intrinsics missing (should not happen after prep).
  - Batches frames for UniDepth inference.
  - Flow/occlusion is stubbed; hook up your flow model if desired.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from unidepth.models import UniDepthV2

# Helpers reused from prep
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from scripts.data_prep.prepare_hypersim_unidepth import sobel_edges, downsample


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Prepared data root (contains scene/cam folders with manifest.jsonl and rgb_full).",
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        required=True,
        help="Output root to write teacher depth/uncert, lows, guidance, and updated manifests.",
    )
    ap.add_argument("--model-id", default="lpiccinelli/unidepth-v2-vitb14", help="UniDepthV2 checkpoint ID.")
    ap.add_argument("--device", default=None, help="torch device (default cuda if available).")
    ap.add_argument("--batch-size", type=int, default=4, help="Frames per forward pass.")
    ap.add_argument("--low-factor", type=int, default=2, choices=[1, 2, 4], help="Downsample factor for SSM branch.")
    ap.add_argument("--save-flow", action="store_true", help="If set, compute low-res flow/occlusion (stub).")
    ap.add_argument("--flow-size", type=int, default=256, help="Low-res size for flow (longer side).")
    ap.add_argument("--stride", type=int, default=1, help="Process every Nth frame from the manifest.")
    return ap.parse_args()


def load_manifest(scene_dir: Path) -> List[Dict]:
    mpath = scene_dir / "manifest.jsonl"
    if not mpath.exists():
        raise FileNotFoundError(f"manifest.jsonl not found in {scene_dir}")
    entries = []
    with open(mpath, "r") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def save_manifest(scene_dir: Path, entries: List[Dict]):
    mpath = scene_dir / "manifest.jsonl"
    mpath.parent.mkdir(parents=True, exist_ok=True)
    with open(mpath, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


def load_rgb(path: Path) -> torch.Tensor:
    from PIL import Image

    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.uint8)
    return torch.from_numpy(arr).permute(2, 0, 1).float()  # [3,H,W]


def run_unidepth(model, device, batch: torch.Tensor, K_batch: torch.Tensor):
    # batch: [B,3,H,W] uint8 float tensor; K_batch: [B,3,3]
    rgb = batch.to(device)
    camera = K_batch.to(device)
    # Force model to keep current resolution (avoid internal resize that mismatches camera)
    _, _, H, W = rgb.shape
    npix = H * W
    model.shape_constraints["pixels_min"] = npix
    model.shape_constraints["pixels_max"] = npix
    aspect = W / H
    model.shape_constraints["ratio_bounds"] = (aspect, aspect)
    with torch.no_grad():
        preds = model.infer(rgb, camera=camera, normalize=True)
    depth = preds["depth"].cpu()  # [B,1,H,W]
    uncert = preds["confidence"].cpu() if "confidence" in preds else None
    intr_out = preds.get("intrinsics")  # [B,3,3]
    camera_prompt = preds.get("camera_prompt") if isinstance(preds, dict) else None
    return depth, uncert, intr_out.cpu() if intr_out is not None else None, camera_prompt


def prepare_K(entry: Dict) -> torch.Tensor:
    K = np.array(
        [
            [entry["intrinsics_full"]["fx"], 0.0, entry["intrinsics_full"]["cx"]],
            [0.0, entry["intrinsics_full"]["fy"], entry["intrinsics_full"]["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return torch.from_numpy(K)


def resize_to_depth_grid(rgb: torch.Tensor, depth_shape: Tuple[int, int]) -> torch.Tensor:
    _, H, W = rgb.shape
    Ht, Wt = depth_shape
    if (H, W) == (Ht, Wt):
        return rgb
    rgb = rgb.unsqueeze(0)
    rgb = F.interpolate(rgb, size=(Ht, Wt), mode="bilinear", align_corners=False)
    return rgb.squeeze(0)


def save_numpy(path: Path, arr: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)


def save_png(path: Path, arr: np.ndarray):
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def compute_flow_stub(prev_rgb_low: torch.Tensor, rgb_low: torch.Tensor):
    # Placeholder: returns zeros
    if prev_rgb_low is None:
        return None, None
    B, C, H, W = rgb_low.shape
    flow = torch.zeros(B, 2, H, W)
    occ = torch.zeros(B, 1, H, W)
    return flow, occ


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = UniDepthV2.from_pretrained(args.model_id).to(device).eval()
    # Avoid internal upscaling that mismatches camera/grid; keep input resolution
    model.shape_constraints["pixels_min"] = 0
    model.shape_constraints["pixels_max"] = 1e9
    model.resolution_level = 0

    scenes = sorted([p for p in args.data_root.iterdir() if p.is_dir()])
    if not scenes:
        raise RuntimeError(f"No scenes found under {args.data_root}")

    for scene in scenes:
        entries = load_manifest(scene)
        if args.stride > 1:
            entries = [e for e in entries if e["frame_index"] % args.stride == 0]
        out_scene_root = args.out_root / scene.name
        out_entries = []

        # batching
        batch_rgb: List[torch.Tensor] = []
        batch_indices: List[int] = []
        batch_entry_refs: List[Dict] = []

        pbar = tqdm(entries, desc=f"[Scene] {scene.name}")
        prev_rgb_low = None

        for entry in pbar:
            rgb_path = args.data_root / entry["rgb_path"]
            rgb = load_rgb(rgb_path)  # [3,H,W]

            batch_rgb.append(rgb)
            batch_indices.append(entry["frame_index"])
            batch_entry_refs.append(entry)

            if len(batch_rgb) >= args.batch_size:
                batch_rgb_t = torch.stack(batch_rgb, dim=0)
                K_batch = torch.stack([prepare_K(e) for e in batch_entry_refs], dim=0)
                depth, uncert, intr_out_batch, cam_prompt = run_unidepth(model, device, batch_rgb_t, K_batch)

                for i in range(len(batch_rgb)):
                    e = batch_entry_refs[i]
                    idx = batch_indices[i]
                    depth_i = depth[i]
                    uncert_i = uncert[i] if uncert is not None else None

                    # Downsample RGB to depth grid if needed
                    rgb_depth_grid = resize_to_depth_grid(batch_rgb[i], depth_i.shape[-2:])

                    # Low-res teacher
                    depth_low = downsample(depth_i, args.low_factor)
                    uncert_low = downsample(uncert_i, args.low_factor) if uncert_i is not None else None

                    # Guidance maps
                    gray = (0.2989 * rgb_depth_grid[0] + 0.5870 * rgb_depth_grid[1] + 0.1140 * rgb_depth_grid[2]).unsqueeze(0)
                    edges_rgb = sobel_edges(gray)
                    edges_depth = sobel_edges(depth_i)
                    gradmag = edges_rgb

                    # Flow/occlusion (stub)
                    flow_low = occ_low = None
                    if args.save_flow:
                        flow_low, occ_low = compute_flow_stub(prev_rgb_low, depth_low)  # placeholder
                    prev_rgb_low = depth_low if args.save_flow else prev_rgb_low

                    # Paths
                    frame_id = f"{idx:06d}"
                    base = out_scene_root / e.get("cam_id", "") if e.get("cam_id") else out_scene_root
                    depth_full_path = base / "depth_teacher_full" / f"{frame_id}.npy"
                    uncert_full_path = base / "uncert_teacher_full" / f"{frame_id}.npy"
                    depth_low_path = base / "depth_teacher_low" / f"{frame_id}.npy"
                    uncert_low_path = base / "uncert_teacher_low" / f"{frame_id}.npy"
                    rgb_grid_path = base / "rgb_depth_grid" / f"{frame_id}.png"
                    edges_rgb_path = base / "edges_rgb" / f"{frame_id}.npy"
                    edges_depth_path = base / "edges_depth" / f"{frame_id}.npy"
                    gradmag_path = base / "gradmag" / f"{frame_id}.npy"
                    flow_path = base / "flow_low" / f"{frame_id}.npy"
                    occ_path = base / "occ_low" / f"{frame_id}.npy"

                    save_numpy(depth_full_path, depth_i.squeeze(0).numpy().astype(np.float32))
                    if uncert_i is not None:
                        save_numpy(uncert_full_path, uncert_i.squeeze(0).numpy().astype(np.float32))
                    save_numpy(depth_low_path, depth_low.squeeze(0).numpy().astype(np.float32))
                    if uncert_low is not None:
                        save_numpy(uncert_low_path, uncert_low.squeeze(0).numpy().astype(np.float32))
                    save_png(rgb_grid_path, rgb_depth_grid.permute(1, 2, 0).byte().numpy())
                    save_numpy(edges_rgb_path, edges_rgb.squeeze(0).numpy().astype(np.float32))
                    save_numpy(edges_depth_path, edges_depth.squeeze(0).numpy().astype(np.float32))
                    save_numpy(gradmag_path, gradmag.squeeze(0).numpy().astype(np.float32))
                    if flow_low is not None and occ_low is not None:
                        save_numpy(flow_path, flow_low.numpy().astype(np.float32))
                        save_numpy(occ_path, occ_low.numpy().astype(np.float32))

                    # Update manifest entry
                    e_out = dict(e)
                    e_out.update(
                        {
                            "depth_teacher_path": str(depth_full_path.relative_to(args.out_root)),
                            "depth_teacher_low_path": str(depth_low_path.relative_to(args.out_root)),
                            "uncert_teacher_path": str(uncert_full_path.relative_to(args.out_root)) if uncert_i is not None else None,
                            "uncert_teacher_low_path": str(uncert_low_path.relative_to(args.out_root)) if uncert_low is not None else None,
                            "rgb_depth_grid_path": str(rgb_grid_path.relative_to(args.out_root)),
                            "edges_rgb_path": str(edges_rgb_path.relative_to(args.out_root)),
                            "edges_depth_path": str(edges_depth_path.relative_to(args.out_root)),
                            "gradmag_path": str(gradmag_path.relative_to(args.out_root)),
                            "low_factor_teacher": args.low_factor,
                            "intrinsics_unidepth_out": intr_out_batch[i].numpy().tolist() if intr_out_batch is not None else None,
                            "camera_prompt": cam_prompt[i] if cam_prompt is not None else None,
                        }
                    )
                    if flow_low is not None and occ_low is not None:
                        e_out.update(
                            {
                                "flow_low_path": str(flow_path.relative_to(args.out_root)),
                                "occ_low_path": str(occ_path.relative_to(args.out_root)),
                            }
                        )
                    out_entries.append(e_out)

                batch_rgb.clear()
                batch_indices.clear()
                batch_entry_refs.clear()

        # process leftover
        if batch_rgb:
            batch_rgb_t = torch.stack(batch_rgb, dim=0)
            K_batch = torch.stack([prepare_K(e) for e in batch_entry_refs], dim=0)
            depth, uncert, intr_out_batch, cam_prompt = run_unidepth(model, device, batch_rgb_t, K_batch)

            for i in range(len(batch_rgb)):
                e = batch_entry_refs[i]
                idx = batch_indices[i]
                depth_i = depth[i]
                uncert_i = uncert[i] if uncert is not None else None

                rgb_depth_grid = resize_to_depth_grid(batch_rgb[i], depth_i.shape[-2:])
                depth_low = downsample(depth_i, args.low_factor)
                uncert_low = downsample(uncert_i, args.low_factor) if uncert_i is not None else None

                gray = (0.2989 * rgb_depth_grid[0] + 0.5870 * rgb_depth_grid[1] + 0.1140 * rgb_depth_grid[2]).unsqueeze(0)
                edges_rgb = sobel_edges(gray)
                edges_depth = sobel_edges(depth_i)
                gradmag = edges_rgb

                flow_low = occ_low = None
                if args.save_flow:
                    flow_low, occ_low = compute_flow_stub(None, depth_low)

                frame_id = f"{idx:06d}"
                base = out_scene_root / e.get("cam_id", "") if e.get("cam_id") else out_scene_root
                depth_full_path = base / "depth_teacher_full" / f"{frame_id}.npy"
                uncert_full_path = base / "uncert_teacher_full" / f"{frame_id}.npy"
                depth_low_path = base / "depth_teacher_low" / f"{frame_id}.npy"
                uncert_low_path = base / "uncert_teacher_low" / f"{frame_id}.npy"
                rgb_grid_path = base / "rgb_depth_grid" / f"{frame_id}.png"
                edges_rgb_path = base / "edges_rgb" / f"{frame_id}.npy"
                edges_depth_path = base / "edges_depth" / f"{frame_id}.npy"
                gradmag_path = base / "gradmag" / f"{frame_id}.npy"
                flow_path = base / "flow_low" / f"{frame_id}.npy"
                occ_path = base / "occ_low" / f"{frame_id}.npy"

                save_numpy(depth_full_path, depth_i.squeeze(0).numpy().astype(np.float32))
                if uncert_i is not None:
                    save_numpy(uncert_full_path, uncert_i.squeeze(0).numpy().astype(np.float32))
                save_numpy(depth_low_path, depth_low.squeeze(0).numpy().astype(np.float32))
                if uncert_low is not None:
                    save_numpy(uncert_low_path, uncert_low.squeeze(0).numpy().astype(np.float32))
                save_png(rgb_grid_path, rgb_depth_grid.permute(1, 2, 0).byte().numpy())
                save_numpy(edges_rgb_path, edges_rgb.squeeze(0).numpy().astype(np.float32))
                save_numpy(edges_depth_path, edges_depth.squeeze(0).numpy().astype(np.float32))
                save_numpy(gradmag_path, gradmag.squeeze(0).numpy().astype(np.float32))
                if flow_low is not None and occ_low is not None:
                    save_numpy(flow_path, flow_low.numpy().astype(np.float32))
                    save_numpy(occ_path, occ_low.numpy().astype(np.float32))

                e_out = dict(e)
                e_out.update(
                    {
                        "depth_teacher_path": str(depth_full_path.relative_to(args.out_root)),
                        "depth_teacher_low_path": str(depth_low_path.relative_to(args.out_root)),
                        "uncert_teacher_path": str(uncert_full_path.relative_to(args.out_root)) if uncert_i is not None else None,
                        "uncert_teacher_low_path": str(uncert_low_path.relative_to(args.out_root)) if uncert_low is not None else None,
                        "rgb_depth_grid_path": str(rgb_grid_path.relative_to(args.out_root)),
                        "edges_rgb_path": str(edges_rgb_path.relative_to(args.out_root)),
                        "edges_depth_path": str(edges_depth_path.relative_to(args.out_root)),
                        "gradmag_path": str(gradmag_path.relative_to(args.out_root)),
                        "low_factor_teacher": args.low_factor,
                        "intrinsics_unidepth_out": intr_out_batch[i].numpy().tolist() if intr_out_batch is not None else None,
                        "camera_prompt": cam_prompt[i] if cam_prompt is not None else None,
                    }
                )
                if flow_low is not None and occ_low is not None:
                    e_out.update(
                        {
                            "flow_low_path": str(flow_path.relative_to(args.out_root)),
                            "occ_low_path": str(occ_path.relative_to(args.out_root)),
                        }
                    )
                out_entries.append(e_out)

            batch_rgb.clear()
            batch_indices.clear()
            batch_entry_refs.clear()

        # save manifest
        save_manifest(out_scene_root, out_entries)
        pbar.close()


if __name__ == "__main__":
    main()
