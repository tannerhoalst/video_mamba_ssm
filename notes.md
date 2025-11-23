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




