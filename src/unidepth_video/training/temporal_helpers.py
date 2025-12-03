"""
Utilities for GT mixing, teacher jitter, and collapse indicators.

Usage (dataset):
    from unidepth_video.training.temporal_helpers import apply_teacher_jitter, choose_gt_mask
    depth_in, used = apply_teacher_jitter(teacher_depth, p_jitter=0.2)

Usage (loss loop):
    gt_mask = choose_gt_mask(batch_has_gt, gt_fraction=0.1)
    L_close = lambda_gt * L_gt + lambda_teacher * L_teacher

Logging collapse:
    mean_abs, frac_small = collapse_indicators(delta_d, tau=1e-4)
"""

import torch
from typing import Tuple


def apply_teacher_jitter(d_teacher: torch.Tensor, p_jitter: float = 0.2, s_min: float = 0.97, s_max: float = 1.03,
                         sigma_rel: float = 0.01) -> Tuple[torch.Tensor, bool]:
    """
    Jitter teacher depth for collapse prevention.
    d_teacher: [..., H, W] in meters.
    Returns (d_in, jitter_applied).
    """
    if torch.rand(1).item() > p_jitter:
        return d_teacher, False

    s = torch.empty(1, device=d_teacher.device).uniform_(s_min, s_max)
    noise_std = sigma_rel * d_teacher.clamp_min(1e-3)
    eps = torch.randn_like(d_teacher) * noise_std
    d_in = s * d_teacher + eps
    return d_in, True


def choose_gt_mask(has_gt: torch.Tensor, gt_fraction: float = 0.1) -> torch.Tensor:
    """
    Select a subset of the batch to use ground-truth supervision.
    has_gt: bool tensor [B] indicating which samples have GT.
    Returns bool mask [B] for GT use.
    """
    probs = torch.full_like(has_gt, gt_fraction, dtype=torch.float)
    probs = probs * has_gt.float()
    return torch.bernoulli(probs).bool()


def collapse_indicators(delta_d: torch.Tensor, tau: float = 1e-4):
    """
    delta_d: predicted residual [B, N, 1, H, W] or similar.
    Returns (mean |ΔD|, fraction |ΔD| < tau).
    """
    abs_d = delta_d.abs()
    mean_abs = abs_d.mean()
    frac_small = (abs_d < tau).float().mean()
    return mean_abs, frac_small


def temporal_mad(seq: torch.Tensor):
    """
    Mean absolute temporal difference for a sequence.
    seq: [B, N, C, H, W]
    Returns scalar tensor.
    """
    diff = (seq[:, 1:] - seq[:, :-1]).abs()
    return diff.mean()


def compare_temporal_mad(refined: torch.Tensor, teacher: torch.Tensor):
    """
    Helps ensure refined isn't identical to teacher temporally.
    Returns ratio of MAD(refined)/MAD(teacher).
    """
    mad_ref = temporal_mad(refined)
    mad_teacher = temporal_mad(teacher).clamp_min(1e-6)
    return (mad_ref / mad_teacher).item()
