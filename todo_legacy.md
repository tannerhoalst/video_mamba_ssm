# VideoMambaDepth-SSM – UniDepthV2 TODO

Long-form video depth via Mamba-style SSM, distilled from **UniDepthV2** as a metric, camera-aware teacher.

---

## Phase 0 – Project Skeleton & Dependencies

- [ ] Create repo structure:
  - [ ] `videomamba_depth/`
  - [ ] `videomamba_depth/models/encoder_student.py`      # student backbone (e.g. DINOv2 / ConvNeXt)
  - [ ] `videomamba_depth/models/teacher_unidepth.py`     # UniDepthV2 wrapper (frozen)
  - [ ] `videomamba_depth/models/temporal_mamba.py`
  - [ ] `videomamba_depth/models/head_depth.py`
  - [ ] `videomamba_depth/models/videomamba_depth.py`
  - [ ] `videomamba_depth/data/dataset_video.py`
  - [ ] `videomamba_depth/data/transforms.py`
  - [ ] `videomamba_depth/losses/metric_depth.py`
  - [ ] `videomamba_depth/losses/midas_se.py`             # optional aux SE loss
  - [ ] `videomamba_depth/losses/temporal_gradient.py`
  - [ ] `videomamba_depth/losses/state_smooth.py`
  - [ ] `videomamba_depth/utils/logger.py`
  - [ ] `videomamba_depth/utils/checkpoint.py`
  - [ ] `videomamba_depth/utils/flow_utils.py`
  - [ ] `videomamba_depth/utils/metrics.py`
  - [ ] `scripts/precompute_teacher_unidepth.py`
  - [ ] `train.py`
  - [ ] `inference_streaming.py`
  - [ ] `config.py` or `configs/` folder

- [ ] Install dependencies:
  - [ ] PyTorch + CUDA
  - [ ] Mamba/VMamba implementation (e.g. `mamba-ssm`, `vmamba`)
  - [ ] `torchvision`, `tqdm`, `einops`, `numpy`
  - [ ] `opencv-python` or `decord` / `pyav` for video I/O
  - [ ] Optical flow lib (RAFT / GMFlow) or placeholder for static-motion masks
  - [ ] UniDepthV2 dependency (e.g. pip install from its repo / wheel)

- [ ] Verify `train.py` skeleton runs:
  - [ ] Parse config
  - [ ] Build dummy model (student encoder + temporal head + depth head)
  - [ ] Run one forward + backward pass on fake data without error

---

## Phase 1 – Teacher & Data Pipeline

### 1.1 UniDepthV2 Teacher Wrapper

- [ ] Implement `videomamba_depth/models/teacher_unidepth.py`:

  - [ ] `class UniDepthTeacher(nn.Module):`
    - [ ] Load pretrained UniDepthV2 weights (e.g. `UniDepthV2.from_pretrained(...)`)
    - [ ] Move to device and set `.eval()`
    - [ ] `for p in self.parameters(): p.requires_grad_(False)`

  - [ ] Implement `predict_depth(self, rgb, K=None)`:
    - Inputs:
      - [ ] `rgb`: `[B,3,H,W]` float tensor, ImageNet-normalized, resized consistently
      - [ ] `K` (optional): `[B,3,3]` camera intrinsics in **resized** coordinates
    - [ ] Forward through UniDepthV2 (either `.infer()` or low-level API)
    - [ ] Return:
      - [ ] `depth_teacher: [B,1,H,W]` (metric depth)
      - [ ] (Optional) `intrinsics_teacher: [B,3,3]`
      - [ ] (Optional) `uncertainty_teacher: [B,1,H,W]`

- [ ] Quick tests:
  - [ ] Run on a single RGB frame (from disk) + nominal intrinsics
  - [ ] Confirm depth shape `[1,1,H,W]`
  - [ ] Inspect value range (metric scale, e.g. meters) and basic sanity (near/far ordering)

### 1.2 Student Encoder (Frame-wise Features)

- [ ] Implement `videomamba_depth/models/encoder_student.py`:

  - [ ] Choose backbone (e.g. DINOv2, ConvNeXt, ResNet, or small ViT)
  - [ ] Implement:
    - [ ] `encode_rgb(rgb: [B,3,H,W]) -> {scale_i: feat_i}`
      - [ ] multi-scale features: `feat_i: [B,C_i,H_i,W_i]`
    - [ ] Optionally expose a global descriptor per frame for keyframe anchoring later

- [ ] Decide:
  - [ ] Whether to initialize from ImageNet-pretrained weights
  - [ ] Whether to freeze early layers initially for stability

- [ ] Test:
  - [ ] Forward random `[B,3,H,W]` and check feature shapes & resolutions

### 1.3 Video Dataset & Clip Sampling (with Intrinsics)

- [ ] Implement `videomamba_depth/data/dataset_video.py`:

  - [ ] Support 2 modes:
    - [ ] Video files (e.g. `.mp4`) → decode into frames
    - [ ] Pre-extracted frame folders

  - [ ] For each video:
    - [ ] Store or compute camera intrinsics (fx, fy, cx, cy or full `K`)
    - [ ] Keep track of original resolution `(H_orig, W_orig)`

  - [ ] `__getitem__`:
    - [ ] Sample a clip of length `N` frames → `[N,3,H_raw,W_raw]`
    - [ ] Retrieve corresponding intrinsics (per video or per frame) → `[N,3,3]`
    - [ ] Apply transforms (see below)
    - [ ] Return:
      - [ ] `rgb_clip: [N,3,H,W]`
      - [ ] `K_clip: [N,3,3]` (resized intrinsics)
      - [ ] `metadata` (video_id, frame indices, original size, etc.)

- [ ] Implement `collate_fn`:
  - [ ] Stack time dimension → `[B,N,3,H,W]`
  - [ ] Stack intrinsics → `[B,N,3,3]`
  - [ ] Collate metadata into a list/dict

- [ ] Test DataLoader:
  - [ ] Iterate and print shapes for several batches
  - [ ] Confirm intrinsics are correctly batched and aligned with frames

### 1.4 Transforms: Resize + Intrinsics Update

- [ ] Implement `videomamba_depth/data/transforms.py`:

  - [ ] `resize_and_update_intrinsics(img, K, target_long_edge)`:
    - [ ] Resize image so the **long edge** = `target_long_edge`
    - [ ] Compute scale factors `sx, sy`
    - [ ] Update intrinsics:
      - [ ] `K'[0,0] = sx * K[0,0]` (fx)
      - [ ] `K'[1,1] = sy * K[1,1]` (fy)
      - [ ] `K'[0,2] = sx * K[0,2]` (cx)
      - [ ] `K'[1,2] = sy * K[1,2]` (cy)
    - [ ] Optionally pad to nearest multiple (e.g. 14/16) and update `cx, cy` for padding

  - [ ] Compose transforms:
    - [ ] Convert to tensor
    - [ ] Resize + pad
    - [ ] ImageNet normalization (mean/std)

  - [ ] Provide a callable that:
    - [ ] Takes `[N, H_raw, W_raw, 3]` + `[N,3,3]`
    - [ ] Returns `[N,3,H,W]` + updated `[N,3,3]`

- [ ] Test:
  - [ ] Feed dummy image & K, verify K changes as expected
  - [ ] Confirm that transformed frames + K are consistent

### 1.5 (Optional) Precompute UniDepthV2 Teacher Depth

- [ ] Implement `scripts/precompute_teacher_unidepth.py`:

  - [ ] Iterate over all videos / frames
  - [ ] Run dataset transforms in **inference mode** (same resizing + K updates)
  - [ ] Call `UniDepthTeacher.predict_depth(rgb, K)` frame-wise
  - [ ] Save:
    - [ ] `depth_teacher` as `.npy` or compressed tensor
    - [ ] (Optional) `uncertainty_teacher` and/or intrinsics snapshot

- [ ] Update `dataset_video.py`:
  - [ ] If `precomputed_depth_root` is set:
    - [ ] Try to load teacher depth per frame
    - [ ] Fall back to on-the-fly teacher if missing
  - [ ] Ensure shapes align with current image resolution

---

## Phase 2 – Temporal Module (Mamba) & Model Wiring

### 2.1 Temporal Tokenization & Reassembly

- [ ] Implement helper functions in `videomamba_depth/models/videomamba_depth.py`:

  - [ ] `temporal_tokenize(feat: [B,N,C,H,W]) -> tokens: [(B*H*W), N, C]`
  - [ ] `temporal_detokenize(tokens: [(B*H*W),N,C], H,W,B) -> [B,N,C,H,W]`

- [ ] Test with random tensors:
  - [ ] `detokenize(tokenize(x))` ≈ `x` (use `allclose`)

### 2.2 Spatial Pre-Mix (Depthwise Conv)

- [ ] Implement `SpatialPreMix` in `videomamba_depth/models/temporal_mamba.py`:

  - [ ] `nn.Conv2d(C, C, kernel_size=3, padding=1, groups=C)` (depthwise)
  - [ ] Forward on `[B*N, C, H, W]`, then reshape back

- [ ] Integrate:
  - [ ] Apply once per scale before temporal tokenization

### 2.3 Mamba Block & Per-Scale Temporal Head

- [ ] Wrap Mamba/SSM implementation:

  - [ ] `class MambaBlock(nn.Module):`
    - [ ] `LayerNorm` over channel dimension
    - [ ] Mamba/SSM core (sequence over time)
    - [ ] Small FFN (Linear → SiLU → Linear)
    - [ ] Residual connections

- [ ] `class TemporalMambaScale(nn.Module):`
  - [ ] Input: `z_seq: [num_tokens, N, C]` (or `[B*H*W, N, C]`)
  - [ ] Apply `num_blocks` `MambaBlock`s sequentially
  - [ ] Output: updated `z_seq` (same shape)
  - [ ] (Later) support hidden states for streaming

### 2.4 Multi-Scale VideoMambaDepth Model

- [ ] Implement `videomamba_depth/models/videomamba_depth.py`:

  - [ ] `__init__`:
    - [ ] `encoder_student` instance
    - [ ] `temporal_heads`:
      - [ ] `spatial_premix` per scale
      - [ ] `TemporalMambaScale` per scale (coarse scales: more blocks; fine scales: fewer)
    - [ ] DPT-style decoder / upsampling head:
      - [ ] Takes per-frame, multi-scale features
      - [ ] Outputs per-frame depth logits/maps

  - [ ] `forward(rgb_clip: [B,N,3,H,W], K_clip: [B,N,3,3], teacher_depth=None)`:
    - [ ] Encode each frame with `encoder_student` → `{scale_i: [B,N,C_i,H_i,W_i]}`
    - [ ] For each scale:
      - [ ] Reshape to `[B*N,C_i,H_i,W_i]`
      - [ ] `SpatialPreMix`
      - [ ] Reshape back `[B,N,C_i,H_i,W_i]`
      - [ ] Tokenize to `[(B*H_i*W_i), N, C_i]`
      - [ ] Add temporal positional encoding on time dimension
      - [ ] Run through `TemporalMambaScale`
      - [ ] Detokenize to `[B,N,C_i,H_i,W_i]`
    - [ ] For each frame `t`:
      - [ ] Feed per-frame multi-scale features into decoder
      - [ ] Get `depth_pred[B,N,1,H_out,W_out]` (same spatial size as input or chosen target)

- [ ] Test:
  - [ ] Forward pass with random `[B,N,3,H,W]`
  - [ ] Confirm shapes and no crashes

---

## Phase 3 – Losses & Training Loop

### 3.1 Metric Depth Distillation (Primary Loss)

- [ ] Implement `videomamba_depth/losses/metric_depth.py`:

  - [ ] `metric_depth_loss(pred, target, mask=None, log_space=True)`:
    - [ ] Option 1: L1/L2 in depth:
      - [ ] `loss = |pred - target|`
    - [ ] Option 2 (recommended): L1 in **log-depth**:
      - [ ] `loss = |log(pred + eps) - log(target + eps)|`
    - [ ] Apply mask if provided (e.g. valid depths / teacher uncertainty threshold)
    - [ ] Average over valid pixels

- [ ] Test:
  - [ ] Create synthetic `target`, `pred = target * noise` and check that loss decreases when `pred` approaches `target`

### 3.2 Optional MiDaS SE (Scale-and-Shift Invariant) Auxiliary Loss

- [ ] Implement `videomamba_depth/losses/midas_se.py`:

  - [ ] `scale_and_shift(pred, target, mask=None)` (LSQ for `a,b`)
  - [ ] `midas_se_loss(pred, target, mask=None)`

- [ ] Use as auxiliary loss:
  - [ ] `L_total = α * L_metric + β * L_se`
  - [ ] Choose `α >> β` so metric supervision dominates

### 3.3 Temporal Gradient Matching (TGM)

- [ ] Implement `videomamba_depth/losses/temporal_gradient.py`:

  - [ ] Compute teacher temporal gradients:
    - [ ] `D_teacher[t+1] - D_teacher[t]`
  - [ ] Compute student temporal gradients:
    - [ ] `D_pred[t+1] - D_pred[t]`
  - [ ] `L_tgm = mean( |ΔD_pred - ΔD_teacher| )` (optionally masked)

### 3.4 State Smoothing Loss

- [ ] Decide representation of `s_t` (per-frame state summary):
  - [ ] e.g. global average pooling of hidden states per scale, concatenated

- [ ] Implement `videomamba_depth/losses/state_smooth.py`:

  - [ ] Input: list/sequence `[s_1, ..., s_N]`
  - [ ] `L_state = mean( ||s_{t+1} - s_t||^2 )` over time
  - [ ] Optionally weight with static-motion masks from optical flow (less penalty on moving regions)

### 3.5 Training Loop with Chunked BPTT & Mixed N

- [ ] Extend `train.py`:

  - [ ] Config parameters:
    - [ ] `clip_length` (max N)
    - [ ] `chunk_size` (N_c) for BPTT
    - [ ] `p_single_frame` (probability of N=1 batches)

  - [ ] Batch handling:
    - [ ] If `N=1`:
      - [ ] Reset temporal states
      - [ ] Forward student encoder + temporal head (degenerate temporal path)
      - [ ] Forward teacher (if not precomputed) for this frame
      - [ ] Compute **metric depth loss** (+ optional SE) only
    - [ ] If `N>1`:
      - [ ] If teacher depth not precomputed:
        - [ ] Run UniDepthTeacher per frame (consider caching per video)
      - [ ] Split along time into chunks of length `N_c`
      - [ ] For each chunk:
        - [ ] Forward chunk with carried hidden states
        - [ ] Compute:
          - [ ] Metric depth loss vs teacher
          - [ ] Temporal gradient loss
          - [ ] State smoothing loss
        - [ ] Accumulate total loss (weighted by `α, β, γ`)
        - [ ] Backprop
        - [ ] Detach hidden states before next chunk (`state = state.detach()`)

  - [ ] Optimizer:
    - [ ] AdamW (or Adam) over student encoder + temporal + decoder
    - [ ] Learning rate scheduler (cosine / step / plateau)

- [ ] Logging:
  - [ ] Track:
    - [ ] Metric depth loss
    - [ ] SE loss (if used)
    - [ ] TGM loss
    - [ ] State smooth loss
  - [ ] Periodically save:
    - [ ] Depth visualizations (teacher vs student)
    - [ ] Temporal slices to inspect flicker

---

## Phase 4 – Streaming Inference & Key-Frame Logic

### 4.1 Hidden-State API for Temporal Head

- [ ] Modify `TemporalMambaScale` to support stateful forward:

  - [ ] `forward(z_seq, state=None) -> (y_seq, new_state)`
    - [ ] If `state is None`: initialize zeros
    - [ ] `state` contains per-block SSM hidden states for that scale

- [ ] Add high-level API to `VideoMambaDepth`:

  - [ ] `init_states() -> dict[scale -> state]`
  - [ ] `step(rgb_frame: [1,3,H,W], K_frame: [1,3,3], states) -> (depth_frame, new_states)`
    - [ ] Encode frame
    - [ ] Run temporal heads with given states
    - [ ] Decode to depth
    - [ ] Return new states

### 4.2 Implement `inference_streaming.py`

- [ ] Implement streaming loop:

  - [ ] Load trained model + weights
  - [ ] `states = model.init_states()`
  - [ ] For each video frame:
    - [ ] Apply same transforms as training (resize + intrinsics update)
    - [ ] Run `depth, states = model.step(frame, K, states)`
    - [ ] Store or display depth map
    - [ ] Optionally convert metric depth to disparity or point cloud

### 4.3 Key-Frame Anchoring & EMA Scale Alignment (Optional)

- [ ] Key-frame storage:
  - [ ] List of:
    - [ ] `(descriptor, states, depth)` for keyframes
    - [ ] `descriptor` = global avg-pooled student feature or teacher depth embedding

- [ ] On every `Δ_k` frames:
  - [ ] Add new keyframe snapshot

- [ ] On each new frame:
  - [ ] Compute current descriptor
  - [ ] Find nearest keyframe via cosine similarity
  - [ ] If similarity > threshold:
    - [ ] Reset current temporal states to keyframe states
    - [ ] Use EMA to align depth scales `(a,b)` if needed:
      - [ ] Update EMA only when state-norm change is small (scene is stable)

---

## Phase 5 – Evaluation & Iteration

- [ ] Select evaluation sequences:
  - [ ] Long VR-like clips (head motion, parallax)
  - [ ] Mix of indoor/outdoor, different FOVs

- [ ] Run baselines:
  - [ ] UniDepthV2 per frame (teacher-only)
  - [ ] VideoMambaDepth-SSM (student)

- [ ] Visual inspection:
  - [ ] Compare flicker, temporal stability, occlusion handling
  - [ ] Compare metric consistency (e.g. same objects at similar distances across frames)

- [ ] Quantitative evaluation (if GT or proxy available):
  - [ ] Per-frame depth error vs GT (or vs teacher)
  - [ ] Temporal variance on static pixels
  - [ ] Flicker metrics / frame-to-frame depth change histograms

- [ ] Hyperparameter tuning:
  - [ ] Loss weights `α_metric`, `β_se`, `γ_tgm`, `δ_state`
  - [ ] Number of Mamba blocks per scale
  - [ ] Chunk size `N_c` and max clip length `N`
  - [ ] Keyframe interval `Δ_k`
  - [ ] EMA thresholds and decay rates

---
