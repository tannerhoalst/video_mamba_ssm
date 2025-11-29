"""
Minimal training loop scaffold to show GT mix, L_close weighting, teacher jitter (already in dataset),
and collapse logging for the temporal refiner. Replace the model/optimizer stubs with your real code.
"""

import argparse
from pathlib import Path

import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from scripts.dataset_video import HypersimWindowDataset, collate_windows
from scripts.models.temporal_refiner import TemporalRefiner
from scripts.training_utils.temporal_helpers import (
    choose_gt_mask,
    collapse_indicators,
    compare_temporal_mad,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--window", type=int, default=12)
    ap.add_argument("--overlap", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--gt-fraction", type=float, default=0.1)
    ap.add_argument("--lambda-gt", type=float, default=1.0)
    ap.add_argument("--lambda-teacher", type=float, default=0.3)
    ap.add_argument("--keyframe-stride", type=int, default=4)
    ap.add_argument("--p-jitter", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    ds = HypersimWindowDataset(
        data_root=args.data_root,
        window=args.window,
        overlap=args.overlap,
        keyframe_stride=args.keyframe_stride,
        p_jitter=args.p_jitter,
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_windows)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = TemporalRefiner().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler()

    for step, batch in enumerate(dl):
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

        # GT mix selection per batch item (uses first frame's has_gt flag for simplicity)
        gt_mask = choose_gt_mask(batch["has_gt"][:, 0], gt_fraction=args.gt_fraction)  # [B]

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=True):
            outputs = model(
                teacher_low=batch["teacher_low"],
                teacher_full=batch["teacher_full"],
                edges_rgb=batch.get("edges_rgb"),
                edges_depth=batch.get("edges_depth"),
            )
            refined = outputs["refined"]
            delta = outputs["delta_low_up"] + outputs["delta_full"]

        # Teacher target
        teacher = batch["teacher_full"]

        # L_teacher (close to teacher)
        l_teacher = (refined - teacher).abs().mean()

        # L_gt only where gt_mask is True
        if any(gt_mask):
            gt_list = batch["depth_gt"]  # list length B; each is list per frame
            gt_stack = []
            for i in range(len(gt_mask)):
                if gt_mask[i]:
                    gt_stack.append(gt_list[i][0].to(device))
                else:
                    gt_stack.append(teacher[i, 0])
            gt = torch.stack(gt_stack)
            l_gt = (refined[:, 0] - gt).abs().mean()
        else:
            l_gt = torch.tensor(0.0, device=teacher.device)

        l_close = args.lambda_gt * l_gt + args.lambda_teacher * l_teacher

        # Collapse indicators
        mean_abs, frac_small = collapse_indicators(delta, tau=1e-4)
        mad_ratio = compare_temporal_mad(refined, teacher)

        scaler.scale(l_close).backward()
        scaler.step(optimizer)
        scaler.update()

        print(
            f"step {step} | L_close {l_close.item():.4f} "
            f"| L_gt {l_gt.item():.4f} | L_teacher {l_teacher.item():.4f} "
            f"| mean|ΔD| {mean_abs.item():.6f} | frac|ΔD|<1e-4 {frac_small.item():.4f} "
            f"| MAD ratio (ref/teacher) {mad_ratio:.4f}"
        )

        if step > 5:  # short demo
            break


if __name__ == "__main__":
    main()
