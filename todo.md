# TODO.md — UniDepthV2 → Video Metric Depth (Temporal + Stereo Ready)
A residual, motion-aware, metric-preserving temporal refiner for VR video depth.

---

# Goal
Extend UniDepthV2 to video by adding a temporal SSM residual smoother that:
- preserves metric scale
- keeps per-frame accuracy
- reduces flicker and aliasing
- becomes a drop-in module for VR stereo warping
Teacher is frozen. Student predicts small residual ΔD only.

---

# Phase 0 — Setup & Baselines
- [x] Verify UniDepthV2 single-frame inference:
  - [x] camera prompt correct
  - [x] pseudo-spherical output mode
  - [x] edge-guided input preprocessing matched
- [x] Implement baseline temporal filters for comparison:
  - [x] EMA
  - [x] temporal bilateral
  - [x] measure static-region variance → flicker baseline
- [ ] Add teacher confidence calibration:
  - [x] record UniDepth uncertainty
  - [ ] capture variance on a static scene
  - [ ] map teacher uncertainty → reliability weighting
- [ ] Install dependencies:
  - [ ] Torch + CUDA, UniDepthV2, Mamba/VMamba, GMFlow/RAFT/UniMatch (optional)
  - [ ] decord or PyAV, einops, tqdm, lpips, ffmpeg
- [ ] Prepare anti-collapse strategy:
  - [ ] small % pseudo-GT clips (KITTI, TUM, ScanNet, synthetic)
  - [ ] OR mild teacher jitter (noise + small scale shift)
- [ ] Define temporal window pattern:
  - [ ] window length N
  - [ ] overlap To
  - [ ] keyframes Tk
  - [ ] keyframe stride Δk
  - [ ] same pattern used for both training and inference

---

# Phase 1 — Precomputation (Teacher Depth + Motion)
- [ ] Create script: scripts/precompute_unidepth_depth.py
  - [ ] decode frames (optional stride)
  - [ ] compute intrinsics for resized inputs
  - [ ] run UniDepthV2 → save depth + uncertainty + camera prompts
  - [ ] downsample RGB to match depth-grid if needed
  - [ ] compute optical flow at LOW RES ONLY (for gating)
  - [ ] compute occlusion masks (fw/bw consistency)
- [ ] Build dataset manifest (.json or .pkl):
  - [ ] frame paths
  - [ ] depth paths
  - [ ] uncertainty
  - [ ] intrinsics
  - [ ] flow paths (optional)
  - [ ] occlusion masks
- [ ] Quick viewer tool:
  - [ ] show RGB / teacher depth / uncertainty
  - [ ] check ranges + continuity

---

# Phase 2 — Dataset & Temporal Window Sampling
- [ ] transforms.py:
  - [ ] resize/pad with intrinsics update
  - [ ] ensure depth scale preserved on resize
  - [ ] generate RGB-low features aligned to depth grid
- [ ] dataset_video.py:
  - [ ] sliding windows of length N
  - [ ] overlap To
  - [ ] insert keyframes Tk spaced by Δk
  - [ ] load depth, uncertainty, RGB-low, flow masks
  - [ ] output shapes:
        depth:  [B, N, 1, Hs, Ws] float32
        rgb:    [B, N, 3, Hr, Wr] float32 in [0,1]
        uncert: [B, N, 1, Hs, Ws] float32
        flow:   [B, N, 2, Hs, Ws] float32 (optional)
- [ ] Add 20% chance temporal reversal augmentation
- [ ] Verify:
  - [ ] shapes correct
  - [ ] zero NaNs
  - [ ] intrinsics scale correct
  - [ ] windows align with keyframes

---

# Phase 3 — Temporal Refiner Model (SSM Residual)
- [ ] Inputs:
  - teacher depth (downsampled)
  - RGB-low (optional)
  - flow-gating mask (optional) shaped [B, N, 1, Hs, Ws], values ∈ [0,1], 0 = ignore temporal loss, 1 = fully trust
  - uncertainty map
  - time encodings
- [ ] Architecture:
  - [ ] Spatial encoder (small CNN) for local patches
  - [ ] Depth-aware spatial mixing:
    - depth-conditioned dilated conv OR
    - shallow bilateral-like conv OR
    - shallow cross-attention (depth keys, rgb queries)
  - [ ] Temporal core:
    - Mamba/VMamba SSM applied per spatial location
    - maintain linear scaling with N
  - [ ] Output:
    - residual ΔD
    - clamp ΔD in forward (soft during training, tunable α/β, depth-aware; see Phase 6 for formula; can tighten at inference if needed)
    - refined depth = teacher + ΔD
- [ ] Mixed precision:
  - [ ] fp16 or bf16
  - [ ] selective fp32 for stability if needed

---

# Phase 4 — Losses (Motion-Aware, No Flow Warp)
- [ ] Teacher-anchored close loss:
  - L1 or L2 on log-depth difference (log(D + ε), ε ≈ 1e-3)
  - weight by calibrated uncertainty
- [ ] Temporal consistency:
  - match ∂(log D)/∂t between refined frames
  - avoid raw-depth gradients (worse stability)
- [ ] Flow/occlusion gating:
  - downweight temporal loss in occluded or very fast-motion regions
- [ ] Residual regularization:
  - small L1 on ΔD magnitude to prevent drift
- [ ] Optional photometric warp:
  - warp RGB using refined depth + intrinsics
  - low weight
  - gated by occlusion
- [ ] Monitor:
  - ratio of L_close : L_temp : L_res
  - ΔD norms per batch
  - any tendency toward ΔD → 0 collapse
  - any oversmoothing

---

# Phase 5 — Training Loop
- [ ] Match inference pattern (window N, overlap To, Tk keyframes, Δk)
- [ ] Random window starts; random resolutions / aspect ratios
- [ ] Optimizer: AdamW; LR schedule: cosine; grad clip; amp on; EMA of weights
- [ ] Checkpoint by temporal metric; log ΔD norms and loss components

---

# Phase 6 — Inference Pipeline
- [ ] inference_smoother.py:
  - [ ] decode frames
  - [ ] compute intrinsics
  - [ ] optionally reuse cached teacher depths/flow
  - [ ] create windows (N, To, Tk, Δk)
  - [ ] run refiner in fp16/bf16
  - [ ] blend overlapping windows with quadratic weighting:
        w = (1 - |pos - 0.5| * 2)^2
  - [ ] apply keyframe influence with temporal decay:
        weight = exp(-Δt / τ)
  - [ ] flow-gating mask expected shape [B, N, 1, Hs, Ws], values ∈ [0,1], 0 = ignore temporal loss, 1 = fully trust
  - [ ] monitor scale drift over long clips (e.g., median depth on static regions); optional light drift penalty/alert
  - [ ] upsample refined depth to full resolution (with intrinsics); optionally edge-aware/guided upsampling (guided by teacher depth or RGB) to preserve edges
  - [ ] clamp ranges
  - [ ] residual clamp configurable (soft during training, optional hard at inference):
        |ΔD| ≤ max(α·D_teacher, β), tunable α/β; loosen when teacher uncertainty is high; tighter in confident/static regions
  - [ ] export:
        per-frame .npy
        16-bit PNG optional
        visualization: teacher vs refined video
- [ ] Use CUDA graphs for speed (optional)

---

# Phase 7 — Evaluation
- [ ] Per-frame metrics (vs pseudo-GT or synthetic):
  - δ thresholds
  - AbsRel
  - RMSE
  - “do-no-harm” vs teacher per-frame metrics
- [ ] Temporal metrics:
  - static-mask MAD variance
  - flow-warped SSIM / LPIPS
  - T-SCIN: LPIPS on temporal gradients
- [ ] Baselines to beat:
  - raw UniDepth
  - EMA
  - temporal bilateral
- [ ] Qualitative:
  - slow-motion depth comparisons
  - edge stability on moving objects
  - scene-wide flicker check
- [ ] Success criteria:
  - lower flicker
  - zero or extremely tiny regression on per-frame metrics

---

# Phase 8 — Stereo & VR Integration
- [ ] Feed refined metric depth to stereo warping (left → right)
- [ ] Validate metric consistency with known baselines
- [ ] Evaluate:
  - reprojection error left→right
  - reprojection error right→left
  - left-right inverse-consistency score
- [ ] Verify:
  - no temporal pop during VR playback
  - no depth-scale drift between windows
  - improved stereo stability vs raw UniDepth

---

# Risks & Notes
- Model may collapse to ΔD→0 without uncertainty weighting or teacher jitter
- Over-smoothing risk in high-motion zones → flow-gated temporal loss is critical
- Never apply global depth rescaling — metric scale is sacred
- Flow should ONLY be gating, never supervision
