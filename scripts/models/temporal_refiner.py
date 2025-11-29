import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

# Require mamba_ssm from the provided .venv
from mamba_ssm import Mamba as MambaBlock


def soft_clamp(delta: torch.Tensor, teacher: torch.Tensor, alpha: float = 0.2, beta: float = 0.05):
    """Soft depth-aware clamp: |ΔD| <= max(alpha * D_teacher, beta) via tanh."""
    limit = torch.maximum(alpha * teacher.abs(), torch.tensor(beta, device=teacher.device, dtype=teacher.dtype))
    return limit * torch.tanh(delta / (limit + 1e-6))


class DepthAwareFiLM(nn.Module):
    """Simple depth-conditioned modulation."""

    def __init__(self, channels: int):
        super().__init__()
        self.gamma = nn.Conv2d(1, channels, kernel_size=1)
        self.beta = nn.Conv2d(1, channels, kernel_size=1)

    def forward(self, x, depth):
        return x * (1 + self.gamma(depth)) + self.beta(depth)


class SpatialEncoder(nn.Module):
    """Light CNN encoder for low-res branch."""

    def __init__(self, in_ch=1, feat_ch=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, feat_ch, 3, padding=1),
            nn.BatchNorm2d(feat_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_ch, feat_ch, 3, padding=1),
            nn.BatchNorm2d(feat_ch),
            nn.ReLU(inplace=True),
        )
        self.film = DepthAwareFiLM(feat_ch)

    def forward(self, x, depth):
        f = self.net(x)
        f = self.film(f, depth)
        return f


class TemporalConvSSM(nn.Module):
    """
    Linear-time temporal mixing using depthwise temporal conv (causal-ish).
    Operates on tokens [B*H*W, C, N].
    """

    def __init__(self, channels: int, kernel_size: int = 5, dropout: float = 0.0):
        super().__init__()
        self.dw = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2, groups=channels)
        self.pw = nn.Conv1d(channels, channels, 1)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x):
        # x: [BHW, C, N]
        y = self.dw(x)
        y = self.act(self.pw(y))
        y = self.drop(y)
        return x + y  # residual


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
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # x: [BHW, N, C]
        y = self.mamba(x)
        y = self.dropout(y)
        return y


class TemporalRefiner(nn.Module):
    def __init__(
        self,
        low_channels: int = 32,
        mid_branch: bool = False,
        mid_factor: int = 2,
        low_factor: int = 2,
        alpha_clamp: float = 0.2,
        beta_clamp: float = 0.05,
        refinement_ch: int = 48,
    ):
        super().__init__()
        self.low_factor = low_factor
        self.mid_branch = mid_branch
        self.mid_factor = mid_factor
        self.alpha_clamp = alpha_clamp
        self.beta_clamp = beta_clamp

        self.enc_low = SpatialEncoder(in_ch=1, feat_ch=low_channels)
        self.temporal = TemporalMambaSSM(low_channels, dropout=0.0)
        self.head_low = nn.Conv2d(low_channels, 1, 3, padding=1)

        if mid_branch:
            self.enc_mid = SpatialEncoder(in_ch=1, feat_ch=low_channels)
            self.temporal_mid = TemporalMambaSSM(low_channels, dropout=0.0)
            self.head_mid = nn.Conv2d(low_channels, 1, 3, padding=1)
            self.fuse_w = nn.Parameter(torch.tensor([0.7, 0.3]))  # low, mid

        # High-res refinement CNN
        # Inputs: teacher_full, upsampled_refined, edges_rgb, edges_depth, optional uncertainty (not wired yet)
        self.refine = nn.Sequential(
            nn.Conv2d(4, refinement_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(refinement_ch, refinement_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(refinement_ch, 1, 3, padding=1),
        )

    def forward(
        self,
        teacher_low: torch.Tensor,   # [B,N,1,Hs,Ws]
        teacher_full: torch.Tensor,  # [B,N,1,Hf,Wf]
        edges_rgb: Optional[torch.Tensor] = None,   # [B,N,1,Hf,Wf]
        edges_depth: Optional[torch.Tensor] = None, # [B,N,1,Hf,Wf]
    ):
        B, N, _, Hs, Ws = teacher_low.shape
        Bf, Nf, _, Hf, Wf = teacher_full.shape
        assert B == Bf and N == Nf

        # Low-res branch
        tl = teacher_low.reshape(B * N, 1, Hs, Ws)
        feat = self.enc_low(tl, tl)  # [B*N, C, Hs, Ws]
        feat = feat.view(B, N, -1, Hs, Ws)                     # [B,N,C,Hs,Ws]
        tokens = feat.permute(0, 3, 4, 1, 2).contiguous()      # [B, Hs, Ws, N, C]
        tokens = tokens.view(-1, N, feat.shape[2])             # [B*Hs*Ws, N, C]
        tokens = self.temporal(tokens)                         # Mamba over time
        feat_out = tokens.view(B, Hs, Ws, N, feat.shape[2]).permute(0, 3, 4, 1, 2).contiguous()  # [B,N,C,Hs,Ws]
        delta_low = self.head_low(feat_out.view(B * N, feat_out.shape[2], Hs, Ws))
        delta_low = delta_low.view(B, N, 1, Hs, Ws)

        # Optional mid branch
        delta_mid_up = 0.0
        if self.mid_branch:
            # Downsample teacher_full to mid resolution
            tm = F.interpolate(teacher_full.view(B * N, 1, Hf, Wf), scale_factor=1.0 / self.mid_factor, mode="bilinear", align_corners=False)
            _, _, Hm, Wm = tm.shape
            featm = self.enc_mid(tm, tm)
            featm = featm.view(B, N, -1, Hm, Wm)                     # [B,N,C,Hm,Wm]
            tokens_m = featm.permute(0, 3, 4, 1, 2).contiguous()     # [B,Hm,Wm,N,C]
            tokens_m = tokens_m.view(-1, N, featm.shape[2])          # [B*Hm*Wm, N, C]
            tokens_m = self.temporal_mid(tokens_m)
            featm_out = tokens_m.view(B, Hm, Wm, N, featm.shape[2]).permute(0, 3, 4, 1, 2).contiguous()  # [B,N,C,Hm,Wm]
            delta_mid = self.head_mid(featm_out.view(B * N, featm_out.shape[2], Hm, Wm))
            delta_mid = delta_mid.view(B, N, 1, Hm, Wm)
            delta_mid_up = F.interpolate(delta_mid.view(B * N, 1, Hm, Wm), size=(Hf, Wf), mode="bilinear", align_corners=False).view(B, N, 1, Hf, Wf)

        # Upsample low-res residual and add to teacher_full
        delta_low_up = F.interpolate(delta_low.view(B * N, 1, Hs, Ws), size=(Hf, Wf), mode="bilinear", align_corners=False).view(B, N, 1, Hf, Wf)
        if self.mid_branch:
            w = torch.softmax(self.fuse_w, dim=0)
            delta_fused = w[0] * delta_low_up + w[1] * delta_mid_up
        else:
            delta_fused = delta_low_up

        # Clamp residual
        delta_fused = soft_clamp(delta_fused, teacher_full, alpha=self.alpha_clamp, beta=self.beta_clamp)

        refined_coarse = teacher_full + delta_fused  # coarse refined at full res

        # High-res refinement
        inputs = [teacher_full, refined_coarse]
        if edges_rgb is not None:
            inputs.append(edges_rgb)
        if edges_depth is not None:
            inputs.append(edges_depth)
        # Ensure we have 4 channels; if missing edges, pad zeros
        while len(inputs) < 4:
            inputs.append(torch.zeros_like(teacher_full))
        ref_in = torch.cat(inputs[:4], dim=2).view(B * N, 4, Hf, Wf)
        delta_full = self.refine(ref_in).view(B, N, 1, Hf, Wf)
        delta_full = soft_clamp(delta_full, teacher_full, alpha=self.alpha_clamp, beta=self.beta_clamp)

        refined = refined_coarse + delta_full
        return {
            "refined": refined,
            "delta_low": delta_low,
            "delta_low_up": delta_low_up,
            "delta_full": delta_full,
        }
