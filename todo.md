# UniDepthV2 + Mamba SSM Residual Smoother – TODO

Goal: Use **UniDepthV2** as the per-frame oracle for metric depth, and train a **Mamba-style SSM** that takes UniDepth depth sequences (plus optional cues) and outputs **small residual corrections** to make depth **more temporally consistent** and slightly more accurate — **without ever drifting far from UniDepth**.

---

## Phase 0 – Project Skeleton & Dependencies

- [ ] Create repo structure:
  - [ ] `videomamba_smoother/`
  - [ ] `videomamba_smoother/models/teacher_unidepth.py`   # UniDepthV2 wrapper (frozen)
  - [ ] `videomamba_smoother/models/ssm_smoother.py`      # Mamba-based temporal smoother
  - [ ] `videomamba_smoother/models/feature_encoder.py`   # small CNN for low-res RGB/aux features (optional)
  - [ ] `videomamba_smoother/data/dataset_video.py`       # video → depth sequence dataset
  - [ ] `videomamba_smoother/data/transforms.py`          # resizing, intrinsics, downsampling
  - [ ] `videomamba_smoother/losses/anchor_to_teacher.py` # L_close: stay near UniDepth
  - [ ] `videomamba_smoother/losses/temporal_smooth.py`   # temporal smoothness / consistency
  - [ ] `videomamba_smoother/losses/utils.py`             # helpers for masks / log-depth
  - [ ] `videomamba_smoother/utils/logger.py`
  - [ ] `videomamba_smoother/utils/checkpoint.py`
  - [ ] `videomamba_smoother/utils/metrics.py`
  - [ ] `scripts/precompute_unidepth_depth.py`
  - [ ] `scripts/preview_depth_sequence.py`
  - [ ] `train_smoother.py`
  - [ ] `inference_smoother.py`
  - [ ] `config.py` or `configs/` folder

- [ ] Install dependencies:
  - [ ] PyTorch + CUDA
  - [ ] Mamba/VMamba implementation (e.g. `mamba-ssm`, `vmamba`)
  - [ ] `torchvision`, `tqdm`, `einops`, `numpy`
  - [ ] `opencv-python` or `decord` / `pyav` for video I/O
  - [ ] UniDepthV2 dependency (install from its repo / wheel)

- [ ] Sanity check:
  - [ ] Confirm a bare `train_smoother.py` runs (argument parsing + dummy loop)
  - [ ] Confirm imports of all modules work

---

## Phase 1 – UniDepthV2 Teacher & Raw Depth Generation

### 1.1 UniDepthV2 Teacher Wrapper

- [ ] Implement / refine `videomamba_smoother/models/teacher_unidepth.py`:

  - [ ] `class UniDepthTeacher(nn.Module):`
    - [ ] Load pretrained UniDepthV2 model (`UniDepthV2.from_pretrained(...)`)
    - [ ] Move to device, set `.eval()`, freeze all params (`requires_grad_(False)`)

  - [ ] `predict_depth(rgb, K=None) -> dict`:
    - [ ] Input:
      - [ ] `rgb`: `[B,3,H,W]` tensor, resized + normalized as UniDepth expects
      - [ ] `K`: optional `[B,3,3]` intrinsics (resized coordinates)
    - [ ] Output dict:
      - [ ] `"depth"`: `[B,1,H,W]` metric depth
      - [ ] `"intrinsics"`: `[B,3,3]` (either given or predicted)
      - [ ] `"uncertainty"`: `[B,1,H,W]` or `None`

- [ ] Quick test script:
  - [ ] Load single RGB frame from disk
  - [ ] Build dummy K
  - [ ] Run `predict_depth`
  - [ ] Visualize depth as a grayscale PNG to inspect sanity

### 1.2 Precompute Teacher Depth Sequences (Offline)

- [ ] Implement `scripts/precompute_unidepth_depth.py`:

  - [ ] CLI args:
    - [ ] `--video_root`, `--output_root`, `--long_edge`, `--stride`, etc.
  - [ ] For each video:
    - [ ] Decode frames at chosen fps/stride
    - [ ] Compute intrinsics for each frame (or per video)
    - [ ] Apply transforms: resize + intrinsics update (see Phase 2)
    - [ ] Run `UniDepthTeacher.predict_depth` on each frame
    - [ ] Save per-frame:
      - [ ] `depth_teacher`: `.npy` or `.pt` → `[1,H,W]`
      - [ ] optional `uncertainty` map
      - [ ] store K and original resolution (per video)

- [ ] Validation:
  - [ ] Write `scripts/preview_depth_sequence.py`:
    - [ ] Load a saved depth sequence
    - [ ] Make a small preview video or grid of frames
    - [ ] Confirm continuity, value range, etc.

---

## Phase 2 – Depth Sequence Dataset & Transforms

### 2.1 Resizing & Intrinsics Handling

- [ ] Implement `videomamba_smoother/data/transforms.py`:

  - [ ] `resize_and_update_intrinsics(img, K, target_long_edge)`:
    - [ ] Resize so long edge = `target_long_edge`
    - [ ] Compute `sx, sy` scaling factors
    - [ ] Update fx, fy, cx, cy in K accordingly
    - [ ] Optionally pad to a multiple (e.g. 16) and adjust cx, cy

  - [ ] `downsample_depth(depth, factor)`:
    - [ ] Use average pooling or bilinear downsample for `[1,H,W] → [1,H',W']`
    - [ ] (Keep metric scale; only change resolution)

  - [ ] If needed: `resize_rgb_for_features(rgb)`:
    - [ ] Low-res RGB for optional feature encoder (e.g. `[3,H/4,W/4]`)

### 2.2 Depth-Sequence Dataset

- [ ] Implement `videomamba_smoother/data/dataset_video.py`:

  - [ ] Dataset uses **precomputed** teacher outputs:
    - [ ] Directory structure, e.g.:
      - [ ] `video_root/<video_id>/frame_<t>.png` or `.jpg`
      - [ ] `depth_root/<video_id>/depth_<t>.npy`
      - [ ] (Optional) `uncertainty_root/<video_id>/uncert_<t>.npy`
      - [ ] JSON/metadata with intrinsics, original size

  - [ ] `__getitem__(idx)`:
    - [ ] Determine which video and temporal window this index corresponds to
    - [ ] Load clip of length `N` frames:
      - [ ] `D_teacher_seq: [N,1,H,W]`
      - [ ] (Optional) low-res `RGB_seq: [N,3,H_rgb,W_rgb]`
      - [ ] `K_seq: [N,3,3]` if needed (mostly for metadata here)
      - [ ] `uncertainty_seq: [N,1,H,W]` (optional)

    - [ ] Downsample depth for SSM if desired:
      - [ ] e.g. `H_s, W_s = H/2, W/2`
      - [ ] `D_teacher_low: [N,1,H_s,W_s]`

    - [ ] Return dict:
      - [ ] `"depth_teacher"`: `[N,1,H_s,W_s]`
      - [ ] `"rgb_low"` (optional): `[N,3,H_rgb,W_rgb]`
      - [ ] `"uncertainty"` (optional): `[N,1,H_s,W_s]`
      - [ ] `"video_id"`, `"frame_indices"` for debugging

  - [ ] Implement `collate_fn`:
    - [ ] Stack sequences into batch:
      - [ ] `[B,N,1,H_s,W_s]` (depth)
      - [ ] `[B,N,3,H_rgb,W_rgb]` if using RGB
      - [ ] `[B,N,1,H_s,W_s]` uncertainty if present

- [ ] DataLoader testing:
  - [ ] Iterate a few batches, print shapes
  - [ ] Confirm no NaNs, shapes match config

---

## Phase 3 – SSM Residual Smoother Model

### 3.1 Optional Feature Encoder (Low-Res RGB / Aux Cues)

- [ ] Implement `videomamba_smoother/models/feature_encoder.py`:

  - [ ] Small CNN (e.g. few Conv + ReLU + pool layers) that:
    - [ ] Takes low-res RGB `[B,N,3,H_rgb,W_rgb]`
    - [ ] Outputs feature maps `[B,N,C_f,H_s,W_s]` aligned with low-res depth grid
  - [ ] Optionally accept:
    - [ ] Concatenated channels of:
      - [ ] depth_teacher
      - [ ] RGB
      - [ ] uncertainty
      - [ ] basic motion cue (e.g. abs difference between frames)

- [ ] Test:
  - [ ] Forward random batch, confirm shapes

### 3.2 Mamba SSM Smoother Core

- [ ] Implement `videomamba_smoother/models/ssm_smoother.py`:

  - [ ] Design decision: operate on **per-spatial-location time series**:
    - [ ] Input depth: `[B,N,1,H_s,W_s]`
    - [ ] Optional features: `[B,N,C_f,H_s,W_s]`
    - [ ] Concatenate along channel: `[B,N,1+C_f,H_s,W_s]`

  - [ ] Flatten spatial dimensions into tokens:
    - [ ] Reshape to `[B, N, C_in, H_s*W_s]`
    - [ ] Permute to `[B * H_s * W_s, N, C_in]` for SSM over time

  - [ ] Mamba block stack:
    - [ ] `class SmootherSSM(nn.Module):`
      - [ ] Several layers of:
        - [ ] LayerNorm over channels
        - [ ] Mamba/SSM core (sequence length = N)
        - [ ] Small FFN
        - [ ] Residual connections

  - [ ] Output residual ΔD:
    - [ ] SSM outputs `[B*H_s*W_s, N, C_out]`
    - [ ] Map to scalar residual per token via Linear → `[B*H_s*W_s, N, 1]`
    - [ ] Reshape back to `[B,N,1,H_s,W_s]`
    - [ ] Final depth:
      - [ ] `D_refined = D_teacher_low + ΔD`

  - [ ] API:
    - [ ] `forward(depth_teacher, feat=None) -> depth_refined, residual`
    - [ ] Where:
      - [ ] `depth_teacher`: `[B,N,1,H_s,W_s]`
      - [ ] `feat` (optional): `[B,N,C_f,H_s,W_s]`

- [ ] Tests:
  - [ ] Forward pass on random data:
    - [ ] Check `D_refined` shape matches `depth_teacher` shape
    - [ ] Ensure no NaNs, gradients flow

---

## Phase 4 – Losses & Training Loop for Smoother

### 4.1 Anchor-to-Teacher Loss (L_close)

- [ ] Implement `videomamba_smoother/losses/anchor_to_teacher.py`:

  - [ ] `anchor_loss(D_refined, D_teacher, uncertainty=None, log_space=True)`:
    - [ ] If `log_space`: use `|log(D_refined + eps) - log(D_teacher + eps)|`
    - [ ] Weight by uncertainty (if available) or use uniform weights
    - [ ] High overall weight (primary guardrail: do NOT deviate too far)

### 4.2 Temporal Smoothness / Consistency Loss (L_temp)

- [ ] Implement `videomamba_smoother/losses/temporal_smooth.py`:

  - [ ] Compute finite differences over time:
    - [ ] `ΔD_refined[t] = D_refined[t+1] - D_refined[t]`
    - [ ] Option A:
      - [ ] Penalize deviation from teacher temporal gradient:
        - [ ] `L_temp = |ΔD_refined - ΔD_teacher|`
    - [ ] Option B:
      - [ ] Penalize flicker directly on refined depth:
        - [ ] `L_temp = |ΔD_refined|` in **static** regions

  - [ ] Use static-motion mask if/when you have optical flow:
    - [ ] Less penalty where motion is large (objects genuinely moving)

### 4.3 Total Loss

- [ ] In `train_smoother.py`:

  - [ ] Compute:
    - [ ] `L_close` from anchor loss (high weight)
    - [ ] `L_temp` from temporal smoothness
    - [ ] Optional regularizers (e.g. smoothness across space)

  - [ ] Combine:
    - [ ] `L_total = λ_close * L_close + λ_temp * L_temp + ...`
    - [ ] Choose λ such that:
      - [ ] `λ_close` dominates (protect per-frame accuracy)
      - [ ] `λ_temp` encourages smoother videos but can’t drag you far from teacher

  - [ ] Run optimizer:
    - [ ] AdamW or Adam over SSM + feature encoder
    - [ ] Learning rate scheduler

- [ ] Training loop checklist:
  - [ ] Load batch from DataLoader
  - [ ] Forward through feature encoder (optional)
  - [ ] Forward through SSM smoother:
    - [ ] `D_refined, residual = model(D_teacher, feat)`
  - [ ] Compute losses, `loss.backward()`
  - [ ] `optimizer.step()`, `optimizer.zero_grad()`
  - [ ] Log metrics every few steps

---

## Phase 5 – Inference Integration & VR Pipeline

### 5.1 Offline Smoother Inference Script

- [ ] Implement `inference_smoother.py`:

  - [ ] Inputs:
    - [ ] `<video_path>`
    - [ ] `--output_depth_path`
    - [ ] `--long_edge`, `--stride`, etc.

  - [ ] Steps:
    - [ ] Decode video frames
    - [ ] Compute intrinsics
    - [ ] Run UniDepthV2 per frame (or load precomputed depth)
    - [ ] Downsample depth to SSM resolution
    - [ ] Form sequences `[N,1,H_s,W_s]`
    - [ ] Run SSM smoother to get `D_refined`
    - [ ] Upsample refined depth back to full resolution (if needed)
    - [ ] Save final refined depth maps:
      - [ ] e.g. one `.npy` per frame or a packed tensor/video

  - [ ] Optional:
    - [ ] Output side-by-side visualization:
      - [ ] Teacher vs Refined depth videos

### 5.2 Hook into VR Stereo Pipeline

- [ ] Decide where in your pipeline you consume depth:
  - [ ] E.g., for right-eye inpainting / reprojection / stereo consistency

- [ ] Replace “raw UniDepth depth” with “SSM-refined depth”
  - [ ] Ensure metric units are preserved
  - [ ] Validate temporal stability in VR playback

---

## Phase 6 – Evaluation & Ablations

- [ ] Collect evaluation clips:
  - [ ] Indoor + outdoor, different motion patterns
  - [ ] Some VR-like camera motions (head bob, parallax)

- [ ] Compare:
  - [ ] UniDepth per frame vs UniDepth + SSM smoother:
    - [ ] Per-frame error vs any available pseudo-GT (if you generate synthetic GT later)
    - [ ] Temporal flicker metrics:
      - [ ] Frame-to-frame variance on static regions
      - [ ] Qualitative: watch depth videos in slow motion

- [ ] Ablations:
  - [ ] Turn off SSM (ΔD = 0) and confirm baseline = UniDepth
  - [ ] Try different λ_close / λ_temp
  - [ ] Try including vs excluding RGB-based features
  - [ ] Try different temporal window sizes N

- [ ] Stop condition:
  - [ ] Accept smoother as “better than or equal to UniDepth” when:
    - [ ] Per-frame metrics are **no worse** than teacher
    - [ ] Temporal flicker visibly reduced on key sequences

---
