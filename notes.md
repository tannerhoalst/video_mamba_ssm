```shell
codex --model=gpt-5.1-codex-max \ 
  --dangerously-bypass-approvals-and-sandbox \
  --enable web_search_request \
  -c model_reasoning_effort="high"


sudo apt-get update
sudo apt-get install -y libjpeg-dev zlib1g-dev libopenjp2-7-dev libtiff-dev libwebp-dev liblcms2-dev
sudo apt-get install -y build-essential cmake ninja-build

python3 -m venv .venv
source .venv/bin/activate

PKG_CONFIG_PATH="/home/thoalst/ffmpeg-8.0/lib/pkgconfig" \
CC="cc -mavx2" \
uv pip install \
  --no-binary av \
  --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  --index-strategy unsafe-best-match \
  -r requirements.txt \
  --upgrade --force-reinstall

uv pip install -v --no-build-isolation -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers
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

python visualize_depth.py /mnt/vrdata/depth_maps/unidepth/ufc.npy

python benchmark_unidepth_inprocess.py --image /home/thoalst/Pictures/Screenshots/ufc.png --runs 60 --warmup 3 --model-id lpiccinelli/unidepth-v2-vitb14

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
python3 video_depth_inference.py \
  --video /mnt/vrdata/clips/sngllittlepucknathan_8kvr265/sngllittlepucknathan_8kvr265_right_flat_2880x2688_h265_clip03.mp4 \
  --output-dir /mnt/vrdata/depth_maps/unidepth/run1 \
  --stack-path /mnt/vrdata/depth_maps/unidepth/run1_stack.npy \
  --batch-size 4 \
  --decoder pyav \
  --hwaccel cuda \
  --hwaccel-output-format cuda

LD_LIBRARY_PATH=/home/thoalst/ffmpeg-8.0/lib:$LD_LIBRARY_PATH \
python3 visualize_depth_video.py \
  /mnt/vrdata/depth_maps/unidepth/run1_stack.npy \
  /mnt/vrdata/depth_maps/unidepth/depth_preview.mp4 \
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