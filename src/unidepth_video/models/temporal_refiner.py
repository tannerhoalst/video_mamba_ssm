from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Require mamba_ssm from the provided .venv
from mamba_ssm import Mamba as MambaBlock


# --------------------------------------------------------------------------- #
# Utility helpers                                                            #
# --------------------------------------------------------------------------- #

def soft_clamp(
    delta: torch.Tensor,
    teacher: torch.Tensor,
    alpha: float = 0.2,
    beta: float = 0.05,
    confidence: Optional[torch.Tensor] = None,
    conf_scale: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Soft depth-aware clamp: |ΔD| <= max(alpha * |D_teacher|, beta) scaled by confidence.
    Returns (clamped_delta, limit_map).
    """
    base_limit = torch.maximum(
        alpha * teacher.abs(),
        torch.tensor(beta, device=teacher.device, dtype=teacher.dtype),
    )
    if confidence is not None:
        limit = base_limit * (1.0 + conf_scale * (1.0 - confidence))
    else:
        limit = base_limit
    clamped = limit * torch.tanh(delta / (limit + 1e-6))
    return clamped, limit


def compute_confidence_from_uncertainty(uncert: torch.Tensor, floor: float = 1e-3):
    """
    Map uncertainty → confidence in [0,1]; monotonically decreasing.
    Using 1 / (1 + uncert) which is stable for fp16.
    """
    return 1.0 / (1.0 + uncert.clamp_min(floor))


def spatial_gradients(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute ∂x and ∂y using Sobel kernels. x: [...,1,H,W]."""
    ky = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], device=x.device, dtype=x.dtype) / 8.0
    kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], device=x.device, dtype=x.dtype) / 8.0
    ky = ky.view(1, 1, 3, 3)
    kx = kx.view(1, 1, 3, 3)
    grad_y = F.conv2d(x, ky, padding=1)
    grad_x = F.conv2d(x, kx, padding=1)
    return grad_x, grad_y


def temporal_sin_cos_encoding(n: int, device, dtype) -> torch.Tensor:
    """
    Sinusoidal scalar per frame in [0,1]; returns [n, 2] (sin, cos).
    """
    idx = torch.arange(n, device=device, dtype=dtype)
    t = idx / max(n - 1, 1)
    return torch.stack([torch.sin(math.pi * t), torch.cos(math.pi * t)], dim=-1)


# --------------------------------------------------------------------------- #
# Building blocks                                                            #
# --------------------------------------------------------------------------- #

class ConvNormAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, d=1, groups=1):
        super().__init__()
        p = (k + (k - 1) * (d - 1) - 1) // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, dilation=d, groups=groups)
        self.norm = nn.BatchNorm2d(out_ch, eps=1e-4)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class DepthAwareMix(nn.Module):
    """
    Optional depth-aware spatial mixing.
    mode = "none" | "dilated" | "bilateral".
    - dilated: depth-conditioned FiLM around a dilated conv.
    - bilateral: shallow guided bilateral-like mixing using depth as guidance.
    """

    def __init__(self, channels: int, mode: str = "none"):
        super().__init__()
        self.mode = mode
        if mode == "dilated":
            self.film = nn.Conv2d(1, channels * 2, 1)
            self.dilated = ConvNormAct(channels, channels, k=3, d=2)
        elif mode == "bilateral":
            self.key = nn.Conv2d(1, channels, 1)  # depth keys
            self.query = nn.Conv2d(channels, channels, 1)
            self.value = nn.Conv2d(channels, channels, 1)
            self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor, depth: torch.Tensor):
        if self.mode == "none":
            return x
        if self.mode == "dilated":
            gamma, beta = torch.chunk(self.film(depth), 2, dim=1)
            y = self.dilated(x)
            return y * (1 + gamma) + beta
        if self.mode == "bilateral":
            k = self.key(depth)
            q = self.query(x)
            v = self.value(x)
            attn = torch.softmax((q * k).mean(dim=1, keepdim=True), dim=-1)
            y = v * attn + x
            return self.proj(y)
        raise ValueError(f"Unknown depth-aware mode {self.mode}")


class SpatialEncoder(nn.Module):
    """Lightweight CNN encoder for the low-res branch."""

    def __init__(self, in_ch: int, feat_ch: int, depth_mode: str):
        super().__init__()
        self.stem = ConvNormAct(in_ch, feat_ch, k=3, s=1)
        self.block = nn.Sequential(
            ConvNormAct(feat_ch, feat_ch, k=3),
            ConvNormAct(feat_ch, feat_ch, k=3),
        )
        self.depth_mix = DepthAwareMix(feat_ch, mode=depth_mode)

    def forward(self, x: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        f = self.stem(x)
        f = self.block(f)
        f = self.depth_mix(f, depth)
        return f


class TemporalMambaSSM(nn.Module):
    """
    Mamba-based temporal mixing per spatial token.
    Expects tokens [B*H*W, N, C].
    """

    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        self.mamba = MambaBlock(
            d_model=channels,
            d_state=16,
            d_conv=4,
            expand=2,
            dt_rank="auto",
        )
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # x: [BHW, N, C]
        y = self.mamba(x)
        y = self.dropout(self.norm(y))
        return y


# --------------------------------------------------------------------------- #
# Main model                                                                 #
# --------------------------------------------------------------------------- #

@dataclass
class TemporalRefinerConfig:
    # Inputs
    include_log_depth: bool = True
    include_confidence: bool = True
    include_gradients: bool = False
    include_time_encoding: bool = True
    include_rgb_edges_low: bool = False
    depth_aware_mode: str = "none"  # "none" | "dilated" | "bilateral"
    # Scales
    low_factor: int = 2
    mid_branch: bool = False
    mid_factor: int = 2
    # Widths
    encoder_ch: int = 48
    temporal_ch: int = 96
    decoder_ch: int = 48
    refinement_ch: int = 64
    # Clamp
    alpha_clamp: float = 0.2
    beta_clamp: float = 0.05
    clamp_conf_scale: float = 0.5
    # Dropout
    dropout: float = 0.0


class TemporalRefiner(nn.Module):
    """
    Temporal residual refiner operating on low-res depth with a Mamba SSM core,
    upsampling to full-res and finishing with a lightweight refinement CNN.
    """

    def __init__(self, cfg: TemporalRefinerConfig = TemporalRefinerConfig()):
        super().__init__()
        self.cfg = cfg

        # Low-res encoder/decoder
        in_ch = 1  # depth
        if cfg.include_log_depth:
            in_ch += 1
        if cfg.include_confidence:
            in_ch += 1
        if cfg.include_gradients:
            in_ch += 2
        if cfg.include_rgb_edges_low:
            in_ch += 1
        if cfg.include_time_encoding:
            in_ch += 2  # sin/cos

        self.enc_low = SpatialEncoder(in_ch=in_ch, feat_ch=cfg.encoder_ch, depth_mode=cfg.depth_aware_mode)
        self.temporal = TemporalMambaSSM(cfg.encoder_ch, dropout=cfg.dropout)
        self.head_low = nn.Sequential(
            ConvNormAct(cfg.encoder_ch, cfg.decoder_ch, k=3),
            nn.Conv2d(cfg.decoder_ch, 1, 3, padding=1),
        )

        if cfg.mid_branch:
            self.enc_mid = SpatialEncoder(in_ch=in_ch, feat_ch=cfg.encoder_ch, depth_mode=cfg.depth_aware_mode)
            self.temporal_mid = TemporalMambaSSM(cfg.encoder_ch, dropout=cfg.dropout)
            self.head_mid = nn.Sequential(
                ConvNormAct(cfg.encoder_ch, cfg.decoder_ch, k=3),
                nn.Conv2d(cfg.decoder_ch, 1, 3, padding=1),
            )
            self.fuse_w = nn.Parameter(torch.tensor([0.7, 0.3]))  # low, mid

        # High-res refinement CNN
        # Inputs: teacher_full, upsampled_refined, optional edges/uncertainty
        refine_in_ch = 2  # teacher_full, coarse
        refine_in_ch += 1  # uncertainty / confidence channel
        refine_in_ch += 2  # edges_rgb, edges_depth placeholders
        self.refine = nn.Sequential(
            ConvNormAct(refine_in_ch, cfg.refinement_ch, k=3),
            ConvNormAct(cfg.refinement_ch, cfg.refinement_ch, k=3),
            nn.Conv2d(cfg.refinement_ch, 1, 3, padding=1),
        )

    # ------------------------------------------------------------------ #
    # Low-res input builder                                              #
    # ------------------------------------------------------------------ #
    def _build_low_inputs(
        self,
        teacher_low: torch.Tensor,
        uncert_low: Optional[torch.Tensor],
        edges_rgb_low: Optional[torch.Tensor],
        time_ids: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Assemble low-res channel stack. Returns (x, confidence_low).
        x: [B,N,C,H,W]
        """
        B, N, _, H, W = teacher_low.shape
        chans = [teacher_low]

        if self.cfg.include_log_depth:
            log_d = torch.log(teacher_low.clamp_min(1e-3))
            chans.append(log_d)

        conf_low = None
        if self.cfg.include_confidence and uncert_low is not None:
            conf_low = compute_confidence_from_uncertainty(uncert_low)
            chans.append(conf_low)

        if self.cfg.include_gradients:
            grad_x, grad_y = spatial_gradients(teacher_low.view(B * N, 1, H, W))
            grad_x = grad_x.view(B, N, 1, H, W)
            grad_y = grad_y.view(B, N, 1, H, W)
            chans.extend([grad_x, grad_y])

        if self.cfg.include_rgb_edges_low and edges_rgb_low is not None:
            chans.append(edges_rgb_low)

        if self.cfg.include_time_encoding:
            if time_ids is None:
                enc = temporal_sin_cos_encoding(N, device=teacher_low.device, dtype=teacher_low.dtype)  # [N,2]
            else:
                # Normalize provided time ids to [0,1]
                tmin = time_ids.min(dim=1, keepdim=True).values
                tmax = time_ids.max(dim=1, keepdim=True).values
                denom = (tmax - tmin).clamp_min(1.0)
                t_norm = (time_ids - tmin) / denom  # [B,N]
                sin = torch.sin(math.pi * t_norm)
                cos = torch.cos(math.pi * t_norm)
                enc = torch.stack([sin, cos], dim=-1)  # [B,N,2]
            enc = enc.view(B, N, 2, 1, 1).expand(B, N, 2, H, W)
            chans.append(enc)

        x = torch.cat(chans, dim=2)
        return x, conf_low

    # ------------------------------------------------------------------ #
    # Forward                                                           #
    # ------------------------------------------------------------------ #
    def forward(
        self,
        teacher_low: torch.Tensor,          # [B,N,1,Hs,Ws]
        teacher_full: torch.Tensor,         # [B,N,1,Hf,Wf]
        uncert_low: Optional[torch.Tensor] = None,   # [B,N,1,Hs,Ws]
        uncert_full: Optional[torch.Tensor] = None,  # [B,N,1,Hf,Wf]
        edges_rgb: Optional[torch.Tensor] = None,    # [B,N,1,Hf,Wf]
        edges_depth: Optional[torch.Tensor] = None,  # [B,N,1,Hf,Wf]
        edges_rgb_low: Optional[torch.Tensor] = None,  # [B,N,1,Hs,Ws]
        time_ids: Optional[torch.Tensor] = None,     # [B,N] optional frame indices
        confidence_low: Optional[torch.Tensor] = None,  # override confidence
    ) -> Dict[str, torch.Tensor]:
        """
        Returns dict with refined depth and residuals:
        refined, delta_low, delta_low_up, delta_full, clamp_hit_rate_low, clamp_hit_rate_full
        """
        B, N, _, Hs, Ws = teacher_low.shape
        _, _, _, Hf, Wf = teacher_full.shape

        # Build low-res stack
        x_low, conf_low_auto = self._build_low_inputs(teacher_low, uncert_low, edges_rgb_low, time_ids)
        conf_low = confidence_low if confidence_low is not None else conf_low_auto

        # Low-res encoder
        feat = self.enc_low(x_low.view(B * N, -1, Hs, Ws), teacher_low.view(B * N, 1, Hs, Ws))
        feat = feat.view(B, N, -1, Hs, Ws)

        # Tokens: [B*Hs*Ws, N, C]
        tokens = feat.permute(0, 3, 4, 1, 2).reshape(-1, N, feat.shape[2])
        tokens = self.temporal(tokens)
        feat_out = tokens.view(B, Hs, Ws, N, feat.shape[2]).permute(0, 3, 4, 1, 2).contiguous()

        delta_low = self.head_low(feat_out.view(B * N, feat_out.shape[2], Hs, Ws)).view(B, N, 1, Hs, Ws)

        # Optional mid branch
        delta_mid_up = None
        if self.cfg.mid_branch:
            tm = F.interpolate(
                teacher_full.view(B * N, 1, Hf, Wf),
                scale_factor=1.0 / self.cfg.mid_factor,
                mode="bilinear",
                align_corners=False,
            )
            Hm, Wm = tm.shape[-2:]
            # Downsample stack to mid
            xm = F.interpolate(x_low.view(B * N, -1, Hs, Ws), size=(Hm, Wm), mode="bilinear", align_corners=False)
            featm = self.enc_mid(xm, tm)
            featm = featm.view(B, N, -1, Hm, Wm)
            tokens_m = featm.permute(0, 3, 4, 1, 2).reshape(-1, N, featm.shape[2])
            tokens_m = self.temporal_mid(tokens_m)
            featm_out = tokens_m.view(B, Hm, Wm, N, featm.shape[2]).permute(0, 3, 4, 1, 2).contiguous()
            delta_mid = self.head_mid(featm_out.view(B * N, featm_out.shape[2], Hm, Wm)).view(B, N, 1, Hm, Wm)
            delta_mid_up = F.interpolate(
                delta_mid.view(B * N, 1, Hm, Wm), size=(Hf, Wf), mode="bilinear", align_corners=False
            ).view(B, N, 1, Hf, Wf)

        # Upsample low-res residual
        delta_low_up = F.interpolate(
            delta_low.view(B * N, 1, Hs, Ws), size=(Hf, Wf), mode="bilinear", align_corners=False
        ).view(B, N, 1, Hf, Wf)

        if delta_mid_up is not None:
            w = torch.softmax(self.fuse_w, dim=0)
            delta_fused_raw = w[0] * delta_low_up + w[1] * delta_mid_up
        else:
            delta_fused_raw = delta_low_up

        # Confidence map at full res
        conf_full = None
        if conf_low is not None:
            conf_full = F.interpolate(conf_low.view(B * N, 1, Hs, Ws), size=(Hf, Wf), mode="bilinear", align_corners=False).view(B, N, 1, Hf, Wf)
        elif uncert_full is not None:
            conf_full = compute_confidence_from_uncertainty(uncert_full)

        # Clamp residual (coarse)
        delta_fused, limit_coarse = soft_clamp(
            delta_fused_raw,
            teacher_full,
            alpha=self.cfg.alpha_clamp,
            beta=self.cfg.beta_clamp,
            confidence=conf_full,
            conf_scale=self.cfg.clamp_conf_scale,
        )
        clamp_hit_rate_low = (delta_fused_raw.abs() > (limit_coarse + 1e-6)).float().mean()

        refined_coarse = teacher_full + delta_fused

        # High-res refinement input assembly
        inputs = [teacher_full, refined_coarse]
        if uncert_full is not None:
            inputs.append(uncert_full)
        elif conf_full is not None:
            inputs.append(1.0 - conf_full)
        else:
            inputs.append(torch.zeros_like(teacher_full))

        if edges_rgb is not None:
            inputs.append(edges_rgb)
        else:
            inputs.append(torch.zeros_like(teacher_full))

        if edges_depth is not None:
            inputs.append(edges_depth)
        else:
            inputs.append(torch.zeros_like(teacher_full))

        refine_in = torch.cat(inputs, dim=2).view(B * N, -1, Hf, Wf)
        delta_full_raw = self.refine(refine_in).view(B, N, 1, Hf, Wf)
        delta_full, limit_ref = soft_clamp(
            delta_full_raw,
            teacher_full,
            alpha=self.cfg.alpha_clamp,
            beta=self.cfg.beta_clamp,
            confidence=conf_full,
            conf_scale=self.cfg.clamp_conf_scale,
        )
        clamp_hit_rate_full = (delta_full_raw.abs() > (limit_ref + 1e-6)).float().mean()

        refined = refined_coarse + delta_full

        return {
            "refined": refined,
            "delta_low": delta_low,
            "delta_low_up": delta_low_up,
            "delta_full": delta_full,
            "clamp_hit_rate_low": clamp_hit_rate_low,
            "clamp_hit_rate_full": clamp_hit_rate_full,
            "confidence_full": conf_full if conf_full is not None else torch.zeros_like(teacher_full),
        }
