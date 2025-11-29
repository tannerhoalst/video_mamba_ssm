"""
Hypersim dataset loader consuming the prepared manifest for UniDepth + temporal refiner.

Assumptions:
- Data prepared by scripts/data_prep/prepare_hypersim_unidepth.py
- manifest.jsonl lives inside each scene folder under --data-root
- Paths in manifest are relative to --data-root

Features:
- Loads full-res RGB, full-res teacher/GT depth, low-res depth, edges, intrinsics, mapping.
- Optional teacher jitter for collapse prevention (applied to model input depth only).
- Exposes has_gt flag and depth_gt_path for GT-aware sampling upstream.

Note:
- Temporal window sampling is left to the sampler/collate; this dataset yields per-frame items.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import torch
from PIL import Image

from scripts.training_utils.temporal_helpers import apply_teacher_jitter


@dataclass
class Sample:
    rgb: torch.Tensor                # [3, Hf, Wf], float32 in [0,1]
    teacher_full: torch.Tensor       # [1, Hf, Wf], float32 meters
    teacher_low: torch.Tensor        # [1, Hs, Ws], float32 meters
    depth_gt: Optional[torch.Tensor] # [1, Hf, Wf] or None
    has_gt: bool
    intrinsics_full: Dict
    intrinsics_low: Dict
    mapping: Dict                    # contains scale & pad for low→full
    edges_rgb: Optional[torch.Tensor]
    edges_depth: Optional[torch.Tensor]
    meta: Dict                       # raw manifest entry
    jitter_applied: bool


def _load_manifest_files(root: Path) -> List[Dict]:
    manifests = list(root.rglob("manifest.jsonl"))
    if not manifests:
        raise FileNotFoundError(f"No manifest.jsonl under {root}")
    entries: List[Dict] = []
    for mpath in sorted(manifests):
        with open(mpath, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                entries.append(entry)
    if not entries:
        raise RuntimeError(f"Empty manifests under {root}")
    return entries


class HypersimFrameDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root: str | Path,
        p_jitter: float = 0.0,
        jitter_sigma_rel: float = 0.01,
        jitter_s_range: tuple[float, float] = (0.97, 1.03),
        load_edges: bool = True,
    ):
        super().__init__()
        self.root = Path(data_root)
        self.entries = _load_manifest_files(self.root)
        self.p_jitter = p_jitter
        self.jitter_sigma_rel = jitter_sigma_rel
        self.jitter_s_range = jitter_s_range
        self.load_edges = load_edges

    def __len__(self):
        return len(self.entries)

    def _load_rgb(self, rel_path: str) -> torch.Tensor:
        img = Image.open(self.root / rel_path).convert("RGB")
        arr = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        return arr

    def _load_npy(self, rel_path: str) -> torch.Tensor:
        arr = np.load(self.root / rel_path)
        return torch.from_numpy(arr).unsqueeze(0).float()

    def __getitem__(self, idx) -> Sample:
        e = self.entries[idx]

        rgb = self._load_rgb(e["rgb_path"])
        teacher_full = self._load_npy(e["depth_gt_path"])
        teacher_low = self._load_npy(e["depth_low_path"])

        edges_rgb = edges_depth = None
        if self.load_edges:
            edges_rgb = self._load_npy(e["edges_rgb_path"])
            edges_depth = self._load_npy(e["edges_depth_path"])

        depth_gt = teacher_full  # Hypersim GT is the teacher_full here
        has_gt = bool(e.get("has_gt", False))

        # Teacher jitter applied to model input depth; supervise against clean teacher
        teacher_in, jitter_applied = apply_teacher_jitter(
            teacher_low, p_jitter=self.p_jitter,
            s_min=self.jitter_s_range[0], s_max=self.jitter_s_range[1],
            sigma_rel=self.jitter_sigma_rel,
        )

        sample = Sample(
            rgb=rgb,
            teacher_full=teacher_full,
            teacher_low=teacher_in,
            depth_gt=depth_gt if has_gt else None,
            has_gt=has_gt,
            intrinsics_full=e["intrinsics_full"],
            intrinsics_low=e["intrinsics_low"],
            mapping=e["mapping"],
            edges_rgb=edges_rgb,
            edges_depth=edges_depth,
            meta=e,
            jitter_applied=jitter_applied,
        )
        return sample


def collate_frames(batch: List[Sample]) -> Dict[str, torch.Tensor | List]:
    """Simple collate to stack tensors; intrinsics/mapping stay as lists of dicts."""
    out = {
        "rgb": torch.stack([b.rgb for b in batch], dim=0),
        "teacher_full": torch.stack([b.teacher_full for b in batch], dim=0),
        "teacher_low": torch.stack([b.teacher_low for b in batch], dim=0),
        "has_gt": torch.tensor([b.has_gt for b in batch], dtype=torch.bool),
        "depth_gt": [b.depth_gt for b in batch],  # may contain None
        "intrinsics_full": [b.intrinsics_full for b in batch],
        "intrinsics_low": [b.intrinsics_low for b in batch],
        "mapping": [b.mapping for b in batch],
        "meta": [b.meta for b in batch],
        "jitter_applied": torch.tensor([b.jitter_applied for b in batch], dtype=torch.bool),
    }
    if batch[0].edges_rgb is not None:
        out["edges_rgb"] = torch.stack([b.edges_rgb for b in batch], dim=0)
    if batch[0].edges_depth is not None:
        out["edges_depth"] = torch.stack([b.edges_depth for b in batch], dim=0)
    return out


# --------------------------------------------------------------------------- #
# Sliding-window wrapper for temporal training/inference                     #
# --------------------------------------------------------------------------- #

class HypersimWindowDataset(torch.utils.data.Dataset):
    """
    Builds sliding windows over each scene.

    Args:
        data_root: prepared data root
        window: number of frames N
        overlap: overlap To (effective stride = window - overlap)
        drop_tail: if True, drop incomplete tail windows
        p_jitter, jitter_sigma_rel, jitter_s_range: passed to frame dataset
        load_edges: load edge maps
    """

    def __init__(
        self,
        data_root: str | Path,
        window: int,
        overlap: int = 0,
        drop_tail: bool = True,
        p_jitter: float = 0.0,
        jitter_sigma_rel: float = 0.01,
        jitter_s_range: tuple[float, float] = (0.97, 1.03),
        load_edges: bool = True,
        keyframe_stride: Optional[int] = None,
        keyframe_offset: int = 0,
    ):
        super().__init__()
        self.frame_ds = HypersimFrameDataset(
            data_root=data_root,
            p_jitter=p_jitter,
            jitter_sigma_rel=jitter_sigma_rel,
            jitter_s_range=jitter_s_range,
            load_edges=load_edges,
        )
        self.window = window
        self.overlap = overlap
        self.drop_tail = drop_tail
        self.keyframe_stride = keyframe_stride
        self.keyframe_offset = keyframe_offset

        # Build scene -> indices mapping
        by_scene: Dict[str, List[int]] = {}
        for i, e in enumerate(self.frame_ds.entries):
            by_scene.setdefault(e["scene_id"], []).append(i)
        for scene, idxs in by_scene.items():
            idxs.sort(key=lambda k: self.frame_ds.entries[k]["frame_index"])
            by_scene[scene] = idxs

        stride = max(1, window - overlap)
        windows = []
        for scene, idxs in by_scene.items():
            n = len(idxs)
            start = 0
            while start + window <= n:
                windows.append((scene, idxs[start:start + window]))
                start += stride
            if not drop_tail and start < n:
                # take tail window padded/truncated
                tail = idxs[-window:]
                windows.append((scene, tail))
        self.windows = windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        scene, idxs = self.windows[i]
        samples = [self.frame_ds[j] for j in idxs]

        rgb = torch.stack([s.rgb for s in samples], dim=0)                     # [N,3,Hf,Wf]
        teacher_full = torch.stack([s.teacher_full for s in samples], dim=0)   # [N,1,Hf,Wf]
        teacher_low = torch.stack([s.teacher_low for s in samples], dim=0)     # [N,1,Hs,Ws]

        has_gt = torch.tensor([s.has_gt for s in samples], dtype=torch.bool)
        depth_gt = [s.depth_gt for s in samples]

        edges_rgb = edges_depth = None
        if samples[0].edges_rgb is not None:
            edges_rgb = torch.stack([s.edges_rgb for s in samples], dim=0)
        if samples[0].edges_depth is not None:
            edges_depth = torch.stack([s.edges_depth for s in samples], dim=0)

        # Keyframe pattern
        keyframe_mask = torch.zeros(len(samples), dtype=torch.bool)
        if self.keyframe_stride is not None and self.keyframe_stride > 0:
            for k in range(self.keyframe_offset, len(samples), self.keyframe_stride):
                keyframe_mask[k] = True
            # always ensure first frame is a keyframe
            keyframe_mask[0] = True

        out = {
            "rgb": rgb,
            "teacher_full": teacher_full,
            "teacher_low": teacher_low,
            "has_gt": has_gt,
            "depth_gt": depth_gt,
            "intrinsics_full": [s.intrinsics_full for s in samples],
            "intrinsics_low": [s.intrinsics_low for s in samples],
            "mapping": [s.mapping for s in samples],
            "scene_id": scene,
            "frame_indices": [s.meta["frame_index"] for s in samples],
            "jitter_applied": torch.tensor([s.jitter_applied for s in samples], dtype=torch.bool),
            "keyframe_mask": keyframe_mask,
        }
        if edges_rgb is not None:
            out["edges_rgb"] = edges_rgb
        if edges_depth is not None:
            out["edges_depth"] = edges_depth
        return out


def collate_windows(batch: List[Dict]) -> Dict[str, torch.Tensor | List]:
    """Collate for HypersimWindowDataset."""
    out = {
        "rgb": torch.stack([b["rgb"] for b in batch], dim=0),                     # [B,N,3,Hf,Wf]
        "teacher_full": torch.stack([b["teacher_full"] for b in batch], dim=0),
        "teacher_low": torch.stack([b["teacher_low"] for b in batch], dim=0),
        "has_gt": torch.stack([b["has_gt"] for b in batch], dim=0),
        "depth_gt": [b["depth_gt"] for b in batch],
        "intrinsics_full": [b["intrinsics_full"] for b in batch],
        "intrinsics_low": [b["intrinsics_low"] for b in batch],
        "mapping": [b["mapping"] for b in batch],
        "scene_id": [b["scene_id"] for b in batch],
        "frame_indices": [b["frame_indices"] for b in batch],
        "jitter_applied": torch.stack([b["jitter_applied"] for b in batch], dim=0),
        "keyframe_mask": torch.stack([b["keyframe_mask"] for b in batch], dim=0),
    }
    if "edges_rgb" in batch[0]:
        out["edges_rgb"] = torch.stack([b["edges_rgb"] for b in batch], dim=0)
    if "edges_depth" in batch[0]:
        out["edges_depth"] = torch.stack([b["edges_depth"] for b in batch], dim=0)
    return out
