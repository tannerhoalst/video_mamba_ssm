# VideoMambaDepth-SSM Implementation Checklist

---

## Phase 0 – Project Skeleton & Dependencies

- [ ] Create repo structure:
  - [ ] `videomamba_depth/`
  - [ ] `videomamba_depth/models/encoder_da.py`
  - [ ] `videomamba_depth/models/temporal_mamba.py`
  - [ ] `videomamba_depth/models/head_depth.py`
  - [ ] `videomamba_depth/models/videomamba_depth.py`
  - [ ] `videomamba_depth/data/dataset_video.py`
  - [ ] `videomamba_depth/data/transforms.py`
  - [ ] `videomamba_depth/losses/midas_se.py`
  - [ ] `videomamba_depth/losses/temporal_gradient.py`
  - [ ] `videomamba_depth/losses/state_smooth.py`
  - [ ] `videomamba_depth/utils/logger.py`
  - [ ] `videomamba_depth/utils/checkpoint.py`
  - [ ] `videomamba_depth/utils/flow_utils.py`
  - [ ] `videomamba_depth/utils/metrics.py`
  - [ ] `train.py`
  - [ ] `inference_streaming.py`
  - [ ] `config.py` or `configs/` folder

- [ ] Install dependencies:
  - [ ] PyTorch + CUDA
  - [ ] Mamba/VMamba implementation (e.g. `mamba-ssm`, `vmamba`)
  - [ ] `torchvision`, `tqdm`, `einops`
  - [ ] Optical flow lib (RAFT / GMFlow) or placeholder

- [ ] Verify `train.py` runs:
  - [ ] Parse config
  - [ ] Build dummy model
  - [ ] Run one forward + backward pass on fake data without error

---

## Phase 1 – Teacher & Data Pipeline

### 1.1 Depth Anything Teacher Wrapper

- [ ] Implement `models/encoder_da.py`:
  - [ ] Load DA2/DA3 weights
  - [ ] Implement `encode_frame(rgb: [B,3,H,W]) -> {scale_i: feat_i}`
  - [ ] Implement `predict_depth(rgb: [B,3,H,W]) -> depth_teacher [B,1,H,W]`
  - [ ] Set encoder to `requires_grad_(False)`

- [ ] Test teacher:
  - [ ] Run on a sample image → get reasonable depth
  - [ ] Inspect value range / shape

### 1.2 Video Dataset & Clip Sampling

- [ ] Implement `data/dataset_video.py`:
  - [ ] Load video or frame-folder paths
  - [ ] Sample clip of length `N` frames → `[N,3,H,W]`
  - [ ] Support `N=1` mode for single-image batches
  - [ ] Return metadata (video_id, frame indices)

- [ ] Implement `collate_fn`:
  - [ ] Stack batch to shape `[B,N,3,H,W]`

- [ ] Test DataLoader:
  - [ ] Iterate and print shapes for several batches

### 1.3 (Optional) Precompute Teacher Depth

- [ ] Create `scripts/precompute_teacher_depth.py`:
  - [ ] Iterate over dataset
  - [ ] Run DA teacher per frame
  - [ ] Save depth maps to disk (`.npy` or compressed)

- [ ] Update dataset:
  - [ ] If precomputed depth exists → load it
  - [ ] Else → fall back to on-the-fly teacher

---

## Phase 2 – Temporal Module (Mamba) & Model Wiring

### 2.1 Temporal Tokenization & Reassembly

- [ ] Implement helper in `videomamba_depth.py`:

  - [ ] `temporal_tokenize(feat: [B,N,C,H,W]) -> tokens: [(B*H*W), N, C]`
  - [ ] `temporal_detokenize(tokens: [(B*H*W),N,C], H,W,B) -> [B,N,C,H,W]`

- [ ] Test with random tensors:
  - [ ] `detokenize(tokenize(x))` ≈ `x`

### 2.2 Spatial Pre-Mix (Depthwise Conv)

- [ ] Implement `SpatialPreMix` in `temporal_mamba.py`:
  - [ ] Depthwise `Conv2d(C,C,3, padding=1, groups=C)`
  - [ ] Forward works on `[B*N, C, H, W]`

- [ ] Integrate:
  - [ ] Apply this before tokenization per scale

### 2.3 Mamba Block & Per-Scale Temporal Head

- [ ] Wrap Mamba/SSM implementation:

  - [ ] `MambaBlock(C)`:
    - [ ] `LayerNorm`
    - [ ] Mamba/SSM core
    - [ ] Small FFN (e.g. Linear → SiLU → Linear)
    - [ ] Residual connection

- [ ] Implement `TemporalMambaScale`:
  - [ ] Takes `z_seq: [num_tokens, N, C]` (or `[B*H*W, N, C]`)
  - [ ] Applies `num_blocks` `MambaBlock`s sequentially
  - [ ] Returns updated sequence

### 2.4 Multi-Scale VideoMambaDepth Model

- [ ] Implement `models/videomamba_depth.py`:

  - [ ] `__init__`:
    - [ ] Store `encoder_da`
    - [ ] `spatial_premix` per scale
    - [ ] `TemporalMambaScale` per scale:
      - [ ] Coarse scales: `num_blocks=2`
      - [ ] Fine scales: `num_blocks=1`
    - [ ] DPT-style decoder for depth

  - [ ] `forward(rgb_clip: [B,N,3,H,W])`:
    - [ ] Encode all frames with DA2/DA3 → multi-scale `[B,N,C_i,H_i,W_i]`
    - [ ] For each scale:
      - [ ] Flatten to `[B*N,C_i,H_i,W_i]`
      - [ ] `SpatialPreMix`
      - [ ] Reshape to `[B,N,C_i,H_i,W_i]`
      - [ ] Tokenize to `[(B*H_i*W_i),N,C_i]`
      - [ ] Add temporal PE(t)
      - [ ] Run through `TemporalMambaScale`
      - [ ] Detokenize back to `[B,N,C_i,H_i,W_i]`
    - [ ] For each frame `t`, feed per-frame multi-scale features into decoder → `depth[B,N,1,H,W]`

- [ ] Test:
  - [ ] Forward pass with random `[B,N,3,H,W]`
  - [ ] Confirm shapes and no crashes

---

## Phase 3 – Losses & Training Loop

### 3.1 MiDaS SE Loss (Teacher Distillation)

- [ ] Implement `losses/midas_se.py`:
  - [ ] `scale_and_shift(pred, target, mask=None)` (LSQ for `a,b`)
  - [ ] `midas_se_loss(pred, target, mask=None)`

- [ ] Test:
  - [ ] Construct `target = a_true * pred + b_true + noise`
  - [ ] Ensure loss is small when relationship is near-linear

### 3.2 Temporal Gradient Matching (TGM)

- [ ] Implement `losses/temporal_gradient.py`:
  - [ ] Compute teacher depth grad: `D_teacher[t+1] - D_teacher[t]`
  - [ ] Compute student depth grad similarly
  - [ ] Loss = L1 or L2 between gradients (optionally masked)

### 3.3 State Smoothing Loss

- [ ] Decide representation of `s_t`:
  - [ ] e.g. global average over states per frame and per scale, then concat

- [ ] Implement `losses/state_smooth.py`:
  - [ ] Input: list/sequence of `s_t`
  - [ ] Compute `||s_{t+1} - s_t||^2`
  - [ ] Apply static-motion mask from optical flow

### 3.4 Training Loop with Chunked BPTT & Mixed N

- [ ] Extend `train.py`:

  - [ ] Configs:
    - [ ] `clip_length` (max N)
    - [ ] `chunk_size` (N_c)
    - [ ] `p_single_frame`

  - [ ] Batch handling:
    - [ ] If `N=1`:
      - [ ] Reset temporal module state (or skip temporal head)
      - [ ] Forward student + teacher
      - [ ] Compute SE loss only
    - [ ] If `N>1`:
      - [ ] Split along time into chunks of length `N_c`
      - [ ] For each chunk:
        - [ ] Forward with carried hidden states
        - [ ] Compute SE + TGM + state smooth
        - [ ] Backprop
        - [ ] Detach hidden states before next chunk (`stop_grad`)

  - [ ] Optimizer:
    - [ ] AdamW (or Adam) + scheduler

- [ ] Logging:
  - [ ] Track SE, TGM, state losses
  - [ ] Periodically save depth visualizations (student vs teacher)

---

## Phase 4 – Streaming Inference & Key-Frame Logic

### 4.1 Hidden-State API for Temporal Head

- [ ] Modify `TemporalMambaScale`:
  - [ ] `forward(z_seq, state=None) -> (y_seq, new_state)`
  - [ ] If `state is None`: initialize zeros
  - [ ] `state` contains per-block SSM hidden states

- [ ] Add to `VideoMambaDepth`:
  - [ ] `init_states() -> dict[scale -> state]`
  - [ ] `step(rgb_frame, states) -> (depth_frame, new_states)`

### 4.2 Implement `inference_streaming.py`

- [ ] Implement streaming loop:
  - [ ] `states = model.init_states()`
  - [ ] For each frame:
    - [ ] Run `step(frame, states)`
    - [ ] Update states
    - [ ] Store/display depth

### 4.3 Key-Frame Anchoring & EMA Scale Alignment

- [ ] Key-frame storage:
  - [ ] List of `(descriptor, states, depth)` for keyframes
  - [ ] Descriptor = global avg-pooled encoder feature

- [ ] On every `Δ_k` frames:
  - [ ] Add new keyframe snapshot

- [ ] On each new frame:
  - [ ] Compute descriptor
  - [ ] Find nearest keyframe via cosine similarity
  - [ ] If similarity > threshold:
    - [ ] Reset current states to keyframe states
    - [ ] Use EMA to align depth scales `(a,b)`:
      - [ ] Update `(a,b)` only if state-norm change < threshold
      - [ ] Otherwise skip update

---

## Phase 5 – Evaluation & Iteration

- [ ] Choose evaluation sequences (long VR-like clips)

- [ ] Run:
  - [ ] Teacher-only (per-frame DA2/DA3 depth)
  - [ ] VideoMambaDepth-SSM model

- [ ] Compare visually:
  - [ ] Check flicker, temporal stability, occlusion handling

- [ ] Compare quantitatively (if GT or proxy available):
  - [ ] Per-frame depth error vs teacher
  - [ ] Temporal variance on static pixels
  - [ ] Flicker metrics / frame-to-frame changes

- [ ] Tune:
  - [ ] Loss weights `α, β, γ`
  - [ ] Number of Mamba blocks per scale
  - [ ] Keyframe interval `Δ_k`
  - [ ] EMA thresholds and decay

---
