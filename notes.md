```shell
codex --model gpt-5.1-codex-max \
  --dangerously-bypass-approvals-and-sandbox \
  --search 

sudo apt-get update
sudo apt-get install -y libjpeg-dev zlib1g-dev libopenjp2-7-dev libtiff-dev libwebp-dev liblcms2-dev
sudo apt-get install -y build-essential cmake ninja-build

set -e
python3 -m venv .venv
source .venv/bin/activate

PKG_CONFIG_PATH="/home/thoalst/ffmpeg-8.0/lib/pkgconfig" \
CC="cc -mavx2" \
uv pip install \
  --no-binary av \
  --no-binary causal-conv1d \
  --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  --index-strategy unsafe-best-match \
  -r requirements.txt \
  --upgrade --force-reinstall

uv pip install -v --no-build-isolation -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers
uv pip install mamba-ssm[causal-conv1d] --no-build-isolation
uv pip install --upgrade "transformers>=4.40.0" "tokenizers>=0.19"

uv pip install -e UniDepth
cd UniDepth/unidepth/ops/knn
TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0+PTX" uv pip install -v --no-build-isolation .
cd ../extract_patches
TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0+PTX" uv pip install -v --no-build-isolation .
cd ../../../..

python -m unidepth.models.unidepthv2.inference \
  --image /home/thoalst/Pictures/Screenshots/ufc.png \
  --model-id lpiccinelli/unidepth-v2-vitb14 \
  --save-depth /mnt/vrdata/depth_maps/unidepth/ufc.npy

python scripts/visualize_depth.py /mnt/vrdata/depth_maps/unidepth/ufc.npy

python scripts/benchmark_unidepth_inprocess.py --image /home/thoalst/Pictures/Screenshots/ufc.png --runs 60 --warmup 3 --model-id lpiccinelli/unidepth-v2-vitb14

cd /tmp
rm -rf libbluray
git clone https://code.videolan.org/videolan/libbluray.git
cd libbluray
meson setup build --prefix=/usr/local && \
cd build
ninja -j"$(nproc)"
sudo ninja install
sudo ldconfig

LD_LIBRARY_PATH=/home/thoalst/ffmpeg-8.0/lib:$LD_LIBRARY_PATH \
python scripts/video_depth_inference.py \
  --video /mnt/vrdata/clips/sngllittlepucknathan_8kvr265/sngllittlepucknathan_8kvr265_right_flat_2880x2688_h265_clip03.mp4 \
  --output-dir /mnt/vrdata/depth_maps/unidepth/run1 \
  --model-id lpiccinelli/unidepth-v2-vitb14 \
  --batch-size 4 \
  --stride 1 \
  --decoder pyav \
  --hwaccel cuda \
  --hwaccel-output-format cuda \
  --stack-dtype float16 \
  --save-uncertainty \
  --uncert-stack-dtype float16

LD_LIBRARY_PATH=/home/thoalst/ffmpeg-8.0/lib:$LD_LIBRARY_PATH \
python scripts/visualize_depth_video.py \
  /mnt/vrdata/depth_maps/unidepth/run1/sngllittlepucknathan_8kvr265_right_flat_2880x2688_h265_clip03_depth_stack.npy \
  /mnt/vrdata/depth_maps/unidepth/run1/depth_preview.mp4 \
  --use-ffmpeg \
  --ffmpeg-bin /home/thoalst/ffmpeg-8.0/bin/ffmpeg \
  --encoder hevc_nvenc \
  --pix-fmt yuv420p \
  --bitrate 20M \
  --percentiles 1 99 \
  --cmap inferno \
  --video-ref /mnt/vrdata/clips/sngllittlepucknathan_8kvr265/sngllittlepucknathan_8kvr265_right_flat_2880x2688_h265_clip03.mp4
```



• You’ll still get very high quality, because the full-res teacher stays intact; the SSM only adds a small
  residual and then you upsample that residual back to full res. The main trade-off is:

  - What the coarse SSM can’t “see” at low res, it can’t correct (tiny high-frequency flicker/edges). But it
    also can’t damage the teacher’s fine detail; it mostly smooths low/mid-frequency flicker.
  - Upsampling the residual itself (bilinear) is usually fine. If you want extra safety on edges, use edge-
    aware upsampling guided by the teacher (e.g., bilateral guided by depth gradients) so you don’t blur
    discontinuities.
  - If you notice missed fine corrections, you can run the SSM at a slightly higher grid (e.g., 1/4 instead
    of 1/8) or add a very light edge-aware refinement after upsampling.

  Net: the output remains full-res; quality loss is minimal, and the SSM mainly reduces flicker without
  harming the teacher’s per-frame detail.


  • Edge-aware upsampling options (simple to implement):

  1. Joint bilateral upsample (guided by RGB or teacher depth edges)

  - Take the coarse refined depth (Hs×Ws) and bilinearly upsample to full size.
  - Run a joint bilateral filter where the guide is either full-res RGB or the teacher depth:
    D_out(p) = sum_q w_s(p,q)·w_r(guide(p),guide(q))·D_up(q) / sum_q w_s·w_r
    where w_s is a spatial Gaussian, w_r is a range Gaussian over the guide values.

  2. Guided filter

  - Use He et al. guided filter with guide = RGB or teacher depth; fast O(1) box filtering style. Inputs:
    full-res guide, coarse-upsampled depth.

  3. Affinity-based (edge-aware weights from gradients)

  - Compute edge weights from guide gradients: w_x = exp(-|∂guide/∂x|/σ), w_y similarly.
  - Apply a small number of weighted smoothing steps on the upsampled depth using those weights to prevent
    crossing edges.

  Practical pick: start with guided filter (fast, few lines in OpenCV/torchvision equivalents), guide
  by teacher depth or RGB. If you want something even simpler, joint bilateral with a small kernel and
  reasonable σ_color/σ_space works fine.



```shell
python scripts/baseline_temporal_filters.py \
  --input-stack /mnt/vrdata/depth_maps/unidepth/run1/sngllittlepucknathan_8kvr265_right_flat_2880x2688_h265_clip03_depth_stack.npy \
  --output-dir /mnt/vrdata/depth_maps/unidepth/run1/filters_clip03 \
  --save-ema --save-bilateral \
  --ema-alpha 0.8 \
  --bilateral-sigma-rel 0.05 \
  --static-thresh-rel 0.02 \
  --output-dtype float16 \
  --median-sample-frames 64 \
  --median-sample-pixels 2000000

source .venv/bin/activate

python scripts/calibrate_uncertainty.py \
  --depth-stack /mnt/vrdata/depth_maps/unidepth/run1/sngllittlepucknathan_8kvr265_right_flat_2880x2688_h265_clip03_depth_stack.npy \
  --uncert-stack /mnt/vrdata/depth_maps/unidepth/run1/sngllittlepucknathan_8kvr265_right_flat_2880x2688_h265_clip03_uncert_stack.npy \
  --bins 20 \
  --curve-csv /mnt/vrdata/depth_maps/unidepth/run1/uncert_curve.csv \
  --chunk-frames 32

```

thoalst@computer:/mnt/vrdata/video_mamba_depth$ source .venv/bin/activate
(.venv) thoalst@computer:/mnt/vrdata/video_mamba_depth$ python scripts/baseline_temporal_filters.py \
  --input-stack /mnt/vrdata/depth_maps/unidepth/run1/sngllittlepucknathan_8kvr265_right_flat_2880x2688_h265_clip03_depth_stack.npy \
  --output-dir /mnt/vrdata/depth_maps/unidepth/run1/filters_clip03 \
  --save-ema --save-bilateral \
  --ema-alpha 0.8 \
  --bilateral-sigma-rel 0.05 \
  --static-thresh-rel 0.02 \
  --output-dtype float16 \
  --median-sample-frames 64 \
  --median-sample-pixels 2000000
Loaded stack (3123, 2688, 2880), dtype=float16, approx_median_depth=2.6250 (sampled 64 frames / 2000000 pixels)
EMA alpha=0.8
Bilateral sigma=0.131250 (rel=0.05)
Static threshold=0.052500 (rel=0.02)
                                                                                                                                                                                                    
Flicker (mean abs diff frame-to-frame):
  Raw   : global=0.028888  static=0.010602
  EMA   : global=0.013450  static=0.004460
  Bilat : global=0.023503  static=0.007137
Saved EMA to /mnt/vrdata/depth_maps/unidepth/run1/filters_clip03/ema.npy
Saved bilateral to /mnt/vrdata/depth_maps/unidepth/run1/filters_clip03/bilateral.npy
(.venv) thoalst@computer:/mnt/vrdata/video_mamba_depth$ python scripts/calibrate_uncertainty.py \
  --depth-stack /mnt/vrdata/depth_maps/unidepth/run1/sngllittlepucknathan_8kvr265_right_flat_2880x2688_h265_clip03_depth_stack.npy \
  --uncert-stack /mnt/vrdata/depth_maps/unidepth/run1/sngllittlepucknathan_8kvr265_right_flat_2880x2688_h265_clip03_uncert_stack.npy \
  --bins 20 \
  --curve-csv /mnt/vrdata/depth_maps/unidepth/run1/uncert_curve.csv \
  --chunk-frames 32
Depth stack: (3123, 2688, 2880), dtype=float16                                                                                                                                                      
Uncertainty stack: (3123, 2688, 2880), dtype=float16
Variance stats: min=0.000356, max=19.332293, median=0.065516
Uncertainty stats: min=0.662232, max=119.214058, median=1.045826
Recommended k for weight = exp(-k * uncert/median_uncert): k=0.6261, mse=0.1192
Saved uncertainty→variance curve to /mnt/vrdata/depth_maps/unidepth/run1/uncert_curve.csv
(.venv) thoalst@computer:/mnt/vrdata/video_mamba_depth$ 

```shell
./download.py \
  --directory /mnt/vrdata/depth_ground_truth/ml-hypersim/hypersim_subset \
  --contains ai_001_001 --contains ai_001_002 --contains ai_001_003 --contains ai_002_001 --contains ai_003_001 \
  --contains camera_keyframe_positions \
  --contains camera_keyframe_orientations \
  --contains metadata_cameras.csv \
  --contains metadata_scene.csv \
  --contains .color.hdf5 \
  --contains depth_meters.hdf5 \
  --contains scene_cam_00_final_preview \
  --silent
```