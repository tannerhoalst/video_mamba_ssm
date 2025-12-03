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
    - [ ] correlation check: corr(|ΔD|, 1 - c(x)); expect larger residuals mainly where teacher confidence is low
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
- [x] Create script: scripts/precompute_unidepth_depth.py
  - [x] decode frames (optional stride)
  - [x] compute intrinsics for resized inputs (records UniDepth-adjusted intrinsics_out)
  - [x] run UniDepthV2 → save depth + uncertainty + camera prompts
  - [x] downsample RGB to match depth-grid if needed
  - [x] save a low-res (×2 or ×4) version of teacher depth + uncertainty for the SSM path, plus record the scale factor / padding offsets
  - [x] precompute high-frequency guidance maps per frame for refinement/upsampling:
        Sobel/Laplacian edges from RGB, Sobel on teacher depth, optional uncertainty gradients
  - [x] precompute guidance grad-magnitude maps G_t for optional edge-aware upsampling
- [x] Build dataset manifest (.json or .pkl):
  - [x] frame paths
  - [x] depth paths
  - [x] uncertainty
  - [x] intrinsics
- [x] Quick viewer tool:
  - [x] show RGB / teacher depth / uncertainty

---

# Phase 2 — Dataset & Temporal Window Sampling
- [x] transforms.py:
  - [x] resize/pad with intrinsics update
  - [x] ensure depth scale preserved on resize
  - [x] generate RGB-low features aligned to depth grid
  - [x] produce paired (low-res, full-res) depth/uncertainty tensors plus a stored scale map for upsampling residuals
  - [x] store explicit spatial mapping tensor M (scale factors + padding offsets) to align low→full grids during upsample
- [x] dataset_video.py:
  - [x] sliding windows of length N
  - [x] overlap To
  - [x] insert keyframes Tk spaced by Δk
  - [x] load depth, uncertainty, RGB-low
  - [x] output shapes (example):
        depth_low:  [B, N, 1, Hs, Ws] float32 (SSM path)
        depth_full: [B, N, 1, Hf, Wf] float32 (teacher/full-res)
        rgb:        [B, N, 3, Hf, Wf] float32 in [0,1]
        uncert_low: [B, N, 1, Hs, Ws] float32
- [x] Add 20% chance temporal reversal augmentation
- [x] Verify:
  - [x] shapes correct
  - [x] zero NaNs
  - [x] intrinsics scale correct
  - [x] windows align with keyframes (validated on available clips; keyframe mask flips with reversal)

---

# Phase 3 — Temporal Refiner Model (SSM Residual)
  - [x] Inputs:
    - Model input channels (low-res path): depth_low, log_depth_low, c(x), optional ∂x u / ∂y u / optional time encodings; optional RGB-low edges
    - Control signals (not fed to SSM): c(x) for clamp modulation (done); teacher-loss weighting / residual-regularization scaling remain in Phase 4 loss wiring
  - [x] Architecture:
    - [x] Multi-scale path:
      - Stage 1: keep teacher depth at full resolution (no rescale of metric units)
      - Stage 2: downsample teacher depth/uncertainty by 2–4× → SSM operates on low-res sequence
      - Optional Stage 2b (toggle): mid-res (×2) SSM branch; fuse ΔD_low and ΔD_mid with learned weights
      - Stage 3: upsample SSM-refined low-res depth back to full resolution (bilinear for v0)
      - Stage 4: tiny high-res refinement CNN takes [teacher_full, upsampled_refined, optional edges/uncertainty] and predicts δD_full; final depth = upsampled_refined + δD_full
    - [x] Spatial encoder (small CNN) for local patches (low-res branch)
    - [x] Depth-aware spatial mixing (toggle):
      - depth-conditioned dilated conv OR
      - shallow bilateral-like conv (implemented options; cross-attn deferred)
  - [x] Temporal core:
    - Mamba/VMamba SSM applied per spatial location (low-res)
    - maintain linear scaling with N
    - v0 baseline: low-res CNN encoder → Mamba → CNN decoder → ΔD_low (no mid-res branch, no depth-aware mixing enabled by default)
    - Suggested widths for v0: encoder C≈32–48, Mamba hidden≈64–96, decoder symmetric
    - [x] Output:
      - residual ΔD_low at low-res; upsample and add to teacher_full; add δD_full from refinement CNN
      - clamp ΔD in forward (soft during training, tunable α/β, confidence-aware); log clamp hit-rate
      - refined depth = teacher_full + upsampled ΔD_low + δD_full
- [x] Mixed precision:
  - [x] fp16 or bf16 (AMP enabled)
  - [ ] selective fp32 for stability if needed (add if instability appears)

---

# Phase 4 — Losses (no optical-flow supervision; depth-based warps allowed)
- [ ] Teacher-anchored close loss:
  - per-pixel weight w_teacher(x) = c(x)
  - L_teacher = Σ w_teacher |log(D_final + ε) − log(D_teacher + ε)|
  - compute at low-res for SSM output and optionally at full-res for refinement δD_full
  - optional small metric consistency stabilizer: α · ‖D_final − D_teacher‖1 with α≈0.01
- [ ] Temporal consistency:
  - match ∂(log D)/∂t between refined frames (same pixel coords; no warping)
  - small-change mask in metric space: g = log(D_teacher); Δg = g_t − g_{t-1}; apply loss only where |Δg| < τ (e.g., τ = 0.05·g or 5 cm cap) to avoid edges/fast movers dominating
  - confidence-aware temporal weight with occlusion gating: w_temp(x) = m(x) · [ w0 · (1 + k · (1 − c(x))) ], where m(x) ∈ [0,1] gates occlusion/motion; high confidence → smaller w_temp; low confidence → stronger smoothing; occluded/fast-motion regions → w_temp suppressed
  - avoid raw-depth gradients (worse stability)
- [ ] Flow/occlusion gating (training):
  - downweight temporal loss in occluded or very fast-motion regions
  - [ ] optional cheap RGB photometric forward/backward check at low-res for gating only (not supervision)
- [ ] Residual regularization:
  - L_res = Σ λ_res * c(x) * |ΔD| to keep ΔD small in high-confidence zones
  - add zero-mean prior on ΔD per window (small weight) to prevent bias/drift
- [ ] Anti-collapse variance term:
  - low-confidence mask L(x) = 1 if c(x) < c_thr else 0; compute σ_low over ΔD where L=1; add L_var = λ_var · max(0, σ_target − σ_low) to keep variance above σ_target in low-confidence areas
  - [ ] Optional photometric warp:
    - warp RGB using refined depth + intrinsics
    - low weight
    - gated by occlusion
  - [ ] Optional RGB-temporal gradient consistency (tiny weight, occlusion-gated)
- [ ] Optional edge-aware refinement loss (tiny weight) to preserve boundaries in high-motion / edge regions
- [ ] Monitor:
  - ratio of L_close : L_temp : L_res
  - ΔD norms per batch
  - clamp hit-rate (fraction of pixels hitting residual clamp) to catch silent clipping or collapse
  - any tendency toward ΔD → 0 collapse
  - any oversmoothing

---

# Phase 5 — Training Loop
- [ ] Match inference pattern (window N, overlap To, Tk keyframes, Δk)
- [ ] Random window starts; random crops within canonical SSM size (keep camera scaling consistent)
- [ ] Optimizer: AdamW; LR schedule: cosine; grad clip; amp on; EMA of weights
- [ ] Checkpoint by temporal metric; log ΔD norms and loss components

---

# Phase 6 — Inference Pipeline
- [ ] inference_smoother.py:
  - [ ] decode frames
  - [ ] compute intrinsics
  - [ ] optionally reuse cached teacher depths
  - [ ] create windows (N, To, Tk, Δk)
  - [ ] run refiner in fp16/bf16:
        1) downsample teacher depth/uncertainty to SSM resolution
        1b) compute confidence c(x) from uncertainty; (optional v0.1) temporally smooth via EMA c_smooth(t) = η c_smooth(t-1) + (1-η) c_raw(t), η ∈ [0.7, 0.9] (use η≈0.8); feed as channel and as control signal for clamp/loss weights
        2) SSM over low-res sequence → ΔD_low; optional mid-res SSM branch
        3) fuse and upsample residuals using stored mapping M to align grids
        4) high-res refinement CNN adds δD_full using teacher_full + upsampled_refined (+ edges/uncertainty/guidance maps)
  - [ ] blend overlapping windows with quadratic weighting:
        w = (1 - |pos - 0.5| * 2)^2
  - [ ] apply keyframe influence with temporal decay:
        weight = exp(-Δt / τ)
  - [ ] occlusion mask (if used) expected shape [B, N, 1, Hs, Ws], values ∈ [0,1]; at inference use it to gate temporal smoothing/blending (not losses)
  - [ ] monitor scale drift over long clips (e.g., median depth on static regions); optional light drift penalty/alert
  - [ ] static-patch drift alert: fixed patch median depth trend check with thresholded warning
  - [ ] periodic teacher re-anchor for very long clips (reset scale/shift bias on a cadence)
  - [ ] optional: enforce tiny per-window bias penalty to keep ΔD mean ~0 (matches zero-mean prior)
  - [ ] upsample refined depth to full resolution (with intrinsics); optionally edge-aware/guided upsampling (guided by teacher depth or RGB) to preserve edges
  - [ ] clamp ranges
  - [ ] residual clamp configurable (soft during training, optional hard at inference):
        B(x) = max(α·D_teacher(x), β) * (γ + (1 − c(x)));
        high confidence → clamp tight (≈ α·D_teacher), low confidence → loosen up to ~2×
        enforce |ΔD(x)| ≤ B(x); prefer soft clamp (tanh/quadratic) during training
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
  - “do-no-harm” vs teacher per-frame metrics: ΔAbsRel ≈ 0, ΔRMSE ≈ 0 (small improvement ideal)
- [ ] Temporal metrics:
  - static-mask MAD variance
  - scale drift over time plot (median depth on a static patch per window)
  - (optional) warped SSIM / LPIPS without flow dependency
  - T-SCIN: LPIPS on temporal gradients
- [ ] Uncertainty-stratified metrics:
  - bin pixels by confidence (high / medium / low using c(x))
  - report AbsRel, RMSE, temporal variance, LPIPS/SSIM on temporal gradients, average |ΔD| per bin
  - desired: high-confidence bin → ΔD ≈ 0; low-confidence bin → larger but useful corrections that cut flicker
- [ ] Baselines to beat:
  - raw UniDepth
  - EMA
  - temporal bilateral
  - “do-no-harm” scale drift (per-window median depth vs teacher)
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

# Minimal v0 slice (first pass to validate SSM refiner)
- Model: v0 baseline (low-res CNN encoder → Mamba → CNN decoder → ΔD_low), no mid-res branch, no depth-aware mixing, no high-res refinement CNN.
- Inputs: [log D_teacher_low, c(x)] only; optional add later ∂x/∂y/∂t u and edges.
- Clamp: soft tanh clamp with simple B(x)=α·D_teacher+β; log clamp hit-rate.
- Losses: L_teacher (log-depth L1 weighted by c), L_temp (same-pixel ∂t log D with small-change mask, no occlusion gate initially), L_res (λ_res·c·|ΔD|). Skip photometric, edge-aware, variance term in v0.
- Training: fixed canonical size; random window starts; N=12, To=6; AMP, grad clip; log ΔD norm and clamp hit-rate.
- Inference: overlap blend with quadratic weights; no keyframe decay yet; monitor static-patch drift; no re-anchor yet.
- Evaluation: per-frame AbsRel/RMSE/δ; temporal static-patch variance; qualitative slow-mo compare teacher vs refined.

---

# Risks & Mitigation Strategy
- Model may collapse to ΔD→0 without uncertainty weighting or teacher jitter  
  - keep teacher jitter on; weight by confidence; monitor ΔD norm and clamp hit-rate; add small anti-collapse variance term where confidence is low
- Over-smoothing risk in high-motion zones → occlusion-aware temporal weighting is critical  
  - use small-change mask + occlusion/motion gating; lower temporal weight near edges/fast motion; keep high-res refinement loss for boundaries
- Metric drift risk across windows → monitor drift and keep zero-mean prior + light bias penalty  
  - zero-mean ΔD prior per window; light bias penalty; keyframe anchoring + overlap blending; periodic teacher re-anchor on long clips; track median depth on static patches
- Misalignment from low→full upsample  
  - validate mapping M with unit-impulse tests; add guided upsampler conditioned on edges if artifacts appear
- Clamp saturation hiding learning  
  - log clamp hit-rate; use soft clamp during training, tighten only for inference; loosen if saturation >~10–20%
- Mask threshold sensitivity (τ for small-change mask)  
  - depth-dependent τ (e.g., 5% depth capped); sweep τ on val clips to avoid ignoring useful areas or over-penalizing edges
- GT scarcity / domain shift  
  - maintain small GT mix; uncertainty-stratified metrics; early-stop on temporal metrics; add pseudo-static validation set
- SSM underfitting high-frequency temporal cues  
  - keep high-res refinement CNN; optional mid-res SSM branch; add shallow temporal convs at high-res if edge temporal artifacts persist
- Latency/VRAM from overlap+keyframes  
  - benchmark with CUDA graphs; adjust N/To/Tk; provide “fast” mode (fewer keyframes, lighter refinement)
- Never apply global depth rescaling — metric scale is sacred
- Flow should ONLY be gating, never supervision

---

Future extension (optional real-world check):
- Add a tiny TUM RGB-D slice for validation or ~5% GT mix:
  sequences: fr1/desk, fr1/desk2, fr1/xyz, fr1/room, fr3/long_office_household

Possible extensions:
- Optional future: low-res occlusion masks for gating only if proven beneficial (avoid flow-based supervision per VDA findings).
