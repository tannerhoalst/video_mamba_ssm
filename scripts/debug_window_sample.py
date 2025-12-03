"""
Quick debug utility to verify dataset/window transforms.

Usage:
  python debug_window_sample.py --data-root /path/to/prepared_hypersim --window 12 --overlap 6
"""

import argparse
from pathlib import Path
from pprint import pprint

import torch

from unidepth_video.data.dataset_video import HypersimWindowDataset, collate_windows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--window", type=int, default=12)
    ap.add_argument("--overlap", type=int, default=6)
    args = ap.parse_args()

    ds = HypersimWindowDataset(
        data_root=args.data_root,
        window=args.window,
        overlap=args.overlap,
        drop_tail=False,
        load_edges=True,
        keyframe_stride=4,
        keyframe_offset=0,
    )
    print(f"Total windows: {len(ds)}")
    sample = ds[0]
    batch = collate_windows([sample])

    print("\nShapes:")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {tuple(v.shape)} {v.dtype}")
    print("\nKeyframe mask (first window):", batch["keyframe_mask"][0].int().tolist())
    print("Frame indices:", batch["frame_indices"][0])
    print("Scene:", batch["scene_id"][0])
    print("\nIntrinsics full[0]:")
    pprint(batch["intrinsics_full"][0])
    print("Intrinsics low[0]:")
    pprint(batch["intrinsics_low"][0])

    # Basic NaN check
    for name in ["rgb", "teacher_full", "teacher_low", "uncert_full", "uncert_low", "mapping_tensor"]:
        if torch.isnan(batch[name]).any() or torch.isinf(batch[name]).any():
            raise ValueError(f"NaN/Inf detected in {name}")
    print("\nNaN/Inf check passed.")


if __name__ == "__main__":
    main()
