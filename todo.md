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
- [x] Add teacher confidence calibration:
  - [x] record UniDepth uncertainty
  - [x] capture variance on a static scene
  - [x] map teacher uncertainty → reliability weighting
- [x] Install dependencies:
  - [x] Torch + CUDA, UniDepthV2, Mamba/VMamba, GMFlow/RAFT/UniMatch (optional)
  - [x] decord or PyAV, einops, tqdm, lpips, ffmpeg
- [ ] Prepare anti-collapse strategy:
  - [x] Collect GT / pseudo-GT clips (commercial-friendly):
    - [x] Hypersim  (indoor, dense, synthetic, perfect GT)
  - [x] Download a small working subset from each:
    - [x] Hypersim: 3–5 scenes (100–300 frames each)
  - [x] Export aligned RGB + metric depth + intrinsics into:
    - `/mnt/vrdata/depth_ground_truth/hypersim/`
  - [x] Normalize formats to match UniDepth + refiner input:
    - [x] fix a single training/inference size for the SSM refiner (e.g., short side → 518 then center-pad/crop to 518×518) and apply the same resize/pad to RGB + GT depth
    - [x] depth units: confirm Hypersim already meters; store as float32 meters (range sanity check)
    - [x] intrinsics: rescale fx/fy by resize factor and shift cx/cy for any padding so they match the resized pixels UniDepth sees
    - [x] grid alignment: ensure RGB and depth share the exact post-resize grid; if teacher depth is lower-res, downsample RGB to that grid and record the mapping
  - [x] Add GT-aware manifest fields:
    - [x] `has_gt: bool`
    - [x] `depth_gt_path: str | None`
    - [x] optional: `dataset_name`, `scene_id` for debugging
  - [x] Implement small GT mix in loss:
    - [x] sample GT clips in ~5–10% of batches
    - [x] define `L_close = λ_gt * L_gt + λ_teacher * L_teacher` with `λ_gt > λ_teacher`
  - [x] Implement teacher jitter path:
    - [x] in dataset loader: with probability `p_jitter`, create `D_in = s * D_teacher + ε`
    - [x] `s ~ Uniform(0.97, 1.03)`
    - [x] `ε ~ N(0, (σ_rel * D_teacher)^2)`, `σ_rel ≈ 0.01`
    - [x] model input uses `D_in`; `L_close` supervised against clean `D_teacher`
  - [x] Log collapse indicators:
    - [x] mean `|ΔD|` per batch
    - [x] fraction of pixels with `|ΔD| < τ` (tiny threshold)
    - [x] temporal metrics vs raw UniDepth (ensure not identical)
- [ ] Define temporal window pattern:
  - [x] window length N (N = 12)
        - ~0.4s context at 30 fps; good balance of temporal smoothing vs. VRAM/latency
  - [x] overlap To (To = 6)
        - 50% overlap reduces boundary artifacts and matches quadratic blending already in plan
  - [x] keyframes Tk (indices {0, 4, 8} per window)
        - First frame anchored; every 4 frames gives three anchors per window for stability
  - [x] keyframe stride Δk (Δk = 4, offset 0)
        - Simple, regular cadence; same for train/infer
  - [x] same pattern used for both training and inference

---

# Phase 1 — Precomputation (Teacher Depth + Motion)
- [ ] Create script: scripts/precompute_unidepth_depth.py
  - [ ] decode frames (optional stride)
  - [ ] compute intrinsics for resized inputs
  - [ ] run UniDepthV2 → save depth + uncertainty + camera prompts
  - [ ] downsample RGB to match depth-grid if needed
  - [ ] save a low-res (×2 or ×4) version of teacher depth + uncertainty for the SSM path, plus record the scale factor / padding offsets
  - [ ] precompute high-frequency guidance maps per frame for refinement/upsampling:
        Sobel/Laplacian edges from RGB, Sobel on teacher depth, optional uncertainty gradients
  - [ ] precompute guidance grad-magnitude maps G_t for optional edge-aware upsampling
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
  - [ ] produce paired (low-res, full-res) depth/uncertainty tensors plus a stored scale map for upsampling residuals
  - [ ] store explicit spatial mapping tensor M (scale factors + padding offsets) to align low→full grids during upsample
- [ ] dataset_video.py:
  - [ ] sliding windows of length N
  - [ ] overlap To
  - [ ] insert keyframes Tk spaced by Δk
  - [ ] load depth, uncertainty, RGB-low, flow masks
  - [ ] output shapes (example):
        depth_low:  [B, N, 1, Hs, Ws] float32 (SSM path)
        depth_full: [B, N, 1, Hf, Wf] float32 (teacher/full-res)
        rgb:        [B, N, 3, Hf, Wf] float32 in [0,1]
        uncert_low: [B, N, 1, Hs, Ws] float32
        flow:       [B, N, 2, Hs, Ws] float32 (optional)
- [ ] Add 20% chance temporal reversal augmentation
- [ ] Verify:
  - [ ] shapes correct
  - [ ] zero NaNs
  - [ ] intrinsics scale correct
  - [ ] windows align with keyframes

---

# Phase 3 — Temporal Refiner Model (SSM Residual)
- [ ] Inputs:
  - teacher depth (low-res for SSM) + full-res teacher depth
  - RGB-low (optional)
  - flow-gating mask (optional) shaped [B, N, 1, Hs, Ws], values ∈ [0,1], 0 = ignore temporal loss, 1 = fully trust
  - uncertainty map (low-res; optionally full-res for refinement)
  - high-res guidance maps: RGB edges, depth edges, optional uncertainty gradients
  - time encodings
- [ ] Architecture:
  - [ ] Multi-scale path:
    - Stage 1: keep teacher depth at full resolution (no rescale of metric units)
    - Stage 2: downsample teacher depth/uncertainty by 2–4× → SSM operates on low-res sequence
    - Optional Stage 2b: mid-res (×2) SSM branch; fuse ΔD_low and ΔD_mid with learned weights
    - Stage 3: upsample SSM-refined low-res depth back to full resolution (e.g., bilinear or learned upsampler)
    - Stage 4: tiny high-res refinement CNN takes [teacher_full, upsampled_refined, optional edges/uncertainty] and predicts δD_full; final depth = upsampled_refined + δD_full
  - [ ] Spatial encoder (small CNN) for local patches (low-res branch)
  - [ ] Depth-aware spatial mixing:
    - depth-conditioned dilated conv OR
    - shallow bilateral-like conv OR
    - shallow cross-attention (depth keys, rgb queries)
  - [ ] Temporal core:
    - Mamba/VMamba SSM applied per spatial location (low-res)
    - maintain linear scaling with N
  - [ ] Output:
    - residual ΔD_low at low-res; upsample and add to teacher_full; add δD_full from refinement CNN
    - clamp ΔD in forward (soft during training, tunable α/β, depth-aware; see Phase 6 for formula; can tighten at inference if needed)
    - refined depth = teacher_full + upsampled ΔD_low + δD_full
- [ ] Mixed precision:
  - [ ] fp16 or bf16
  - [ ] selective fp32 for stability if needed

---

# Phase 4 — Losses (Motion-Aware, No Flow Warp)
- [ ] Teacher-anchored close loss:
  - L1 or L2 on log-depth difference (log(D + ε), ε ≈ 1e-3)
  - weight by calibrated uncertainty
  - compute at low-res for SSM output and optionally at full-res for refinement δD_full
  - optional small metric consistency stabilizer: α · ‖D_final − D_teacher‖1 with α≈0.01
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
- [ ] Optional RGB-temporal gradient consistency (tiny weight, occlusion-gated)
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
  - [ ] run refiner in fp16/bf16:
        1) downsample teacher depth/uncertainty to SSM resolution
        2) SSM over low-res sequence → ΔD_low; optional mid-res SSM branch
        3) fuse and upsample residuals using stored mapping M to align grids
        4) high-res refinement CNN adds δD_full using teacher_full + upsampled_refined (+ edges/uncertainty/guidance maps)
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

---

Future extension (optional real-world check):
- Add a tiny TUM RGB-D slice for validation or ~5% GT mix:
  sequences: fr1/desk, fr1/desk2, fr1/xyz, fr1/room, fr3/long_office_household
