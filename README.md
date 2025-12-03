# UniDepth Video Temporal Refiner

This repository extends UniDepthV2 with a temporal residual refiner (Mamba-based SSM) for VR video depth.  
Install in editable mode from the repo root:

```bash
pip install -e .
```

Run the training stub (short demo) via console script:

```bash
unidepth-train-refiner --data-root /mnt/vrdata/depth_ground_truth/hypersim --window 12 --overlap 6 --batch-size 1 --device cuda
```

Entry points under `scripts/` (e.g., debug_window_sample.py) also assume the editable install and import the `unidepth_video` package from `src/`.
