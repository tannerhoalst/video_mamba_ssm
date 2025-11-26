"""Convert an uncertainty (and optionally depth) stack to an MP4 preview.

Usage examples
--------------
# Uncertainty only
python visualize_depth_uncertainty_video.py uncert_stack.npy uncert_preview.mp4 --use-ffmpeg --ffmpeg-bin /path/to/ffmpeg

# Side-by-side depth + uncertainty
python visualize_depth_uncertainty_video.py uncert_stack.npy uncert_depth_combo.mp4 \
  --depth-source depth_stack.npy --layout hstack --use-ffmpeg --ffmpeg-bin /path/to/ffmpeg
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import imageio.v2 as imageio
import numpy as np
from matplotlib import colormaps as mpl_cmaps
from tqdm import tqdm


class ArraySequence:
    """Lightweight loader for stacked .npy or directory of per-frame .npy arrays."""

    def __init__(self, source: Path, raw_shape: Optional[Tuple[int, int, int]] = None, raw_dtype: str = "float16"):
        if source.is_dir():
            self.mode = "dir"
            self.files = sorted(source.glob("*.npy"))
            if not self.files:
                raise FileNotFoundError(f"No .npy files found in {source}")
            self.length = len(self.files)
            self.shape = self._peek_shape(np.load(self.files[0], mmap_mode="r"))
        else:
            self.mode = "stack"
            try:
                self.stack = np.load(source, mmap_mode="r")
            except Exception:
                if raw_shape is None:
                    raise
                dtype = np.float16 if raw_dtype == "float16" else np.float32
                self.stack = np.memmap(source, mode="r", dtype=dtype, shape=raw_shape)
            if self.stack.ndim not in (3, 4):
                raise ValueError(f"Expected stacked array of shape (N,H,W) or (N,1,H,W); got {self.stack.shape}")
            if self.stack.ndim == 4:
                if self.stack.shape[1] != 1:
                    raise ValueError(f"4D stack must be (N,1,H,W); got {self.stack.shape}")
                self.stack = self.stack[:, 0]
            self.length = self.stack.shape[0]
            self.shape = self.stack.shape[1:]

    @staticmethod
    def _peek_shape(arr: np.ndarray) -> Tuple[int, int]:
        if arr.ndim == 2:
            return arr.shape
        if arr.ndim == 3 and arr.shape[0] == 1:
            return arr.shape[1:]
        raise ValueError(f"Frame should be 2D; got {arr.shape}")

    def iterate(self, step: int = 1) -> Iterable[np.ndarray]:
        if self.mode == "dir":
            for path in self.files[::step]:
                yield self._to_2d(np.load(path, mmap_mode="r"))
        else:
            for idx in range(0, self.length, step):
                yield self._to_2d(self.stack[idx])

    @staticmethod
    def _to_2d(arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 2:
            return arr
        if arr.ndim == 3 and arr.shape[0] == 1:
            return arr[0]
        raise ValueError(f"Expected 2D array; got {arr.shape}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "uncert_source",
        type=Path,
        help="Stacked .npy (shape N,H,W) or directory with per-frame *_uncert.npy files.",
    )
    parser.add_argument("out", type=Path, help="Output video file (e.g., uncert_preview.mp4).")
    parser.add_argument(
        "--depth-source",
        type=Path,
        help="Optional depth stack/dir to render side-by-side with uncertainty (must match frame count and resolution).",
    )
    parser.add_argument(
        "--layout",
        choices=["hstack", "vstack"],
        default="hstack",
        help="Arrangement when depth is provided: horizontal or vertical stacking (default: hstack).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Output video frame rate. If omitted and --video-ref is provided, fps is probed from that video; otherwise defaults to 24.",
    )
    parser.add_argument(
        "--uncert-cmap",
        default="magma",
        help="Matplotlib colormap for uncertainty visualization (default: magma).",
    )
    parser.add_argument(
        "--depth-cmap",
        default="inferno",
        help="Colormap for depth when --depth-source is used (default: inferno).",
    )
    parser.add_argument(
        "--percentiles",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=(1.0, 99.0),
        help="Percentile range for uncertainty normalization (default: 1 99).",
    )
    parser.add_argument(
        "--depth-percentiles",
        type=float,
        nargs=2,
        default=(1.0, 99.0),
        metavar=("LOW", "HIGH"),
        help="Percentile range for depth normalization when depth is provided (default: 1 99).",
    )
    parser.add_argument(
        "--sample-step",
        type=int,
        default=10,
        help="Use every Nth frame to estimate global percentiles (speeds up long videos).",
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        help="Optional directory to also dump per-frame colored PNGs while creating the video.",
    )
    parser.add_argument(
        "--encoder",
        default="h264_nvenc",
        help="FFmpeg encoder when using --use-ffmpeg (e.g., h264_nvenc, hevc_nvenc, libx264).",
    )
    parser.add_argument(
        "--pix-fmt",
        default="yuv420p",
        help="Output pixel format for ffmpeg (default: yuv420p).",
    )
    parser.add_argument(
        "--bitrate",
        default=None,
        help="Optional bitrate for ffmpeg output (e.g., 10M). If omitted, ffmpeg defaults apply or use --qp instead.",
    )
    parser.add_argument(
        "--qp",
        type=int,
        default=None,
        help="Optional constant QP (e.g., 23) for ffmpeg encoders that support it.",
    )
    parser.add_argument(
        "--ffmpeg-bin",
        default="ffmpeg",
        help="Path to ffmpeg binary (use your hardware-accelerated build).",
    )
    parser.add_argument(
        "--use-ffmpeg",
        action="store_true",
        help="Stream frames to ffmpeg for faster GPU encoding (uses --encoder/--pix-fmt/--bitrate/--qp).",
    )
    parser.add_argument(
        "--video-ref",
        type=Path,
        help="Optional original video to probe fps from (uses ffprobe located next to --ffmpeg-bin).",
    )
    parser.add_argument(
        "--raw-shape",
        type=int,
        nargs=3,
        metavar=("N", "H", "W"),
        help="If the uncertainty stack file is a raw memmap without .npy header, provide the shape (frames, height, width).",
    )
    parser.add_argument(
        "--raw-dtype",
        choices=["float16", "float32"],
        default="float16",
        help="Datatype for --raw-shape fallback (default float16).",
    )
    parser.add_argument(
        "--depth-raw-shape",
        type=int,
        nargs=3,
        metavar=("N", "H", "W"),
        help="Optional raw memmap shape for depth stack if not .npy.",
    )
    parser.add_argument(
        "--depth-raw-dtype",
        choices=["float16", "float32"],
        default="float16",
        help="Datatype for --depth-raw-shape fallback (default float16).",
    )
    return parser.parse_args()


def compute_global_range(seq: ArraySequence, percentiles: Tuple[float, float], sample_step: int) -> Tuple[float, float]:
    lows: List[float] = []
    highs: List[float] = []
    for frame in seq.iterate(step=max(1, sample_step)):
        finite = frame[np.isfinite(frame)]
        if finite.size == 0:
            continue
        lo, hi = np.percentile(finite, percentiles)
        lows.append(float(lo))
        highs.append(float(hi))
    if not lows:
        raise ValueError("No finite values found to compute percentiles.")
    vmin = float(np.median(lows))
    vmax = float(np.median(highs))
    if vmin == vmax:
        vmax = vmin + 1e-6
    return vmin, vmax


def map_to_rgb(arr: np.ndarray, vmin: float, vmax: float, cmap_name: str) -> np.ndarray:
    """Map a 2D frame to uint8 RGB using a colormap."""
    cmap = mpl_cmaps.get_cmap(cmap_name)
    norm = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)
    norm = np.nan_to_num(norm, nan=0.0, posinf=0.0, neginf=0.0)
    rgb_float = cmap(norm)[..., :3]  # drop alpha
    return (rgb_float * 255).astype(np.uint8)


def maybe_write_png(frame_rgb: np.ndarray, idx: int, frames_dir: Optional[Path]) -> None:
    if frames_dir is None:
        return
    frames_dir.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(frames_dir / f"frame{idx:06d}.png", frame_rgb)


def render_video(args: argparse.Namespace) -> None:
    uncert_seq = ArraySequence(
        args.uncert_source, raw_shape=tuple(args.raw_shape) if args.raw_shape else None, raw_dtype=args.raw_dtype
    )
    uncert_vmin, uncert_vmax = compute_global_range(uncert_seq, tuple(args.percentiles), args.sample_step)
    print(f"Uncertainty normalization range: [{uncert_vmin:.4f}, {uncert_vmax:.4f}] using cmap='{args.uncert_cmap}'")

    depth_seq = None
    depth_vmin = depth_vmax = None
    if args.depth_source:
        depth_seq = ArraySequence(
            args.depth_source,
            raw_shape=tuple(args.depth_raw_shape) if args.depth_raw_shape else None,
            raw_dtype=args.depth_raw_dtype,
        )
        if depth_seq.length != uncert_seq.length:
            print(
                f"Warning: depth frames ({depth_seq.length}) != uncertainty frames ({uncert_seq.length}); "
                "using the minimum length."
            )
        if depth_seq.shape != uncert_seq.shape:
            raise ValueError(
                f"Depth shape {depth_seq.shape} does not match uncertainty shape {uncert_seq.shape}; cannot stack."
            )
        depth_vmin, depth_vmax = compute_global_range(depth_seq, tuple(args.depth_percentiles), args.sample_step)
        print(f"Depth normalization range: [{depth_vmin:.4f}, {depth_vmax:.4f}] using cmap='{args.depth_cmap}'")

    fps_value = resolve_fps(args)

    if args.use_ffmpeg:
        run_ffmpeg_render(uncert_seq, depth_seq, args, uncert_vmin, uncert_vmax, depth_vmin, depth_vmax, fps_value)
    else:
        writer = imageio.get_writer(args.out, fps=fps_value, codec="libx264", quality=8)
        length = uncert_seq.length if depth_seq is None else min(uncert_seq.length, depth_seq.length)
        uncert_iter = uncert_seq.iterate()
        depth_iter = depth_seq.iterate() if depth_seq else None
        for idx in tqdm(range(length), desc="Rendering video"):
            uncert = next(uncert_iter)
            uncert_rgb = map_to_rgb(uncert, uncert_vmin, uncert_vmax, args.uncert_cmap)
            if depth_seq:
                depth = next(depth_iter)  # type: ignore
                depth_rgb = map_to_rgb(depth, depth_vmin, depth_vmax, args.depth_cmap)  # type: ignore
                frame_rgb = stack_frames(depth_rgb, uncert_rgb, args.layout)
            else:
                frame_rgb = uncert_rgb
            writer.append_data(frame_rgb)
            maybe_write_png(frame_rgb, idx, args.frames_dir)
        writer.close()
        print(f"Wrote uncertainty preview to: {args.out}")
        if args.frames_dir:
            print(f"Per-frame PNGs stored in: {args.frames_dir}")


def stack_frames(left: np.ndarray, right: np.ndarray, layout: str) -> np.ndarray:
    if layout == "hstack":
        return np.concatenate([left, right], axis=1)
    return np.concatenate([left, right], axis=0)


def build_ffmpeg_cmd(args: argparse.Namespace, width: int, height: int, fps: float) -> list[str]:
    cmd = [
        args.ffmpeg_bin,
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        f"{width}x{height}",
        "-r",
        f"{fps}",
        "-i",
        "-",
        "-an",
        "-c:v",
        args.encoder,
        "-pix_fmt",
        args.pix_fmt,
    ]
    if args.qp is not None:
        cmd += ["-qp", str(args.qp)]
    if args.bitrate:
        cmd += ["-b:v", args.bitrate]
    cmd += [str(args.out)]
    return cmd


def run_ffmpeg_render(
    uncert_seq: ArraySequence,
    depth_seq: Optional[ArraySequence],
    args: argparse.Namespace,
    uncert_vmin: float,
    uncert_vmax: float,
    depth_vmin: Optional[float],
    depth_vmax: Optional[float],
    fps: float,
) -> None:
    base_h, base_w = uncert_seq.shape
    if depth_seq:
        if depth_seq.shape != uncert_seq.shape:
            raise ValueError(
                f"Depth shape {depth_seq.shape} does not match uncertainty shape {uncert_seq.shape}; cannot stack."
            )
        if args.layout == "hstack":
            height, width = base_h, base_w * 2
        else:
            height, width = base_h * 2, base_w
        length = min(uncert_seq.length, depth_seq.length)
        depth_iter = depth_seq.iterate()
    else:
        height, width = base_h, base_w
        length = uncert_seq.length
        depth_iter = None

    cmd = build_ffmpeg_cmd(args, width, height, fps)
    print("Running ffmpeg:", " ".join(shlex.quote(part) for part in cmd))

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None

    try:
        uncert_iter = uncert_seq.iterate()
        for idx in tqdm(range(length), desc="Rendering video"):
            uncert = next(uncert_iter)
            uncert_rgb = map_to_rgb(uncert, uncert_vmin, uncert_vmax, args.uncert_cmap)
            if depth_iter:
                depth = next(depth_iter)
                depth_rgb = map_to_rgb(depth, depth_vmin, depth_vmax, args.depth_cmap)  # type: ignore
                frame_rgb = stack_frames(depth_rgb, uncert_rgb, args.layout)
            else:
                frame_rgb = uncert_rgb
            proc.stdin.write(frame_rgb.astype(np.uint8).tobytes())
            maybe_write_png(frame_rgb, idx, args.frames_dir)
    finally:
        if proc.stdin:
            proc.stdin.close()
            proc.stdin = None
        stderr = proc.stderr.read() if proc.stderr else b""
        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"ffmpeg failed with code {ret}:\n{stderr.decode(errors='ignore')}")
    target = "depth+uncertainty" if depth_seq else "uncertainty"
    print(f"Wrote {target} preview to: {args.out} via ffmpeg ({args.encoder}, {args.pix_fmt})")
    if args.frames_dir:
        print(f"Per-frame PNGs stored in: {args.frames_dir}")


def resolve_fps(args: argparse.Namespace) -> float:
    if args.fps:
        return float(args.fps)
    if args.video_ref:
        probed = probe_fps(args.ffmpeg_bin, args.video_ref)
        if probed:
            return probed
    return 24.0


def probe_fps(ffmpeg_bin: str, video_path: Path) -> Optional[float]:
    ffprobe = str(Path(ffmpeg_bin).with_name("ffprobe"))
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate,avg_frame_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip().splitlines()
    except Exception:
        return None
    for line in out:
        if "/" in line:
            num, den = line.split("/", 1)
            try:
                num_f, den_f = float(num), float(den)
                if den_f != 0:
                    return num_f / den_f
            except Exception:
                continue
        else:
            try:
                return float(line)
            except Exception:
                continue
    return None


if __name__ == "__main__":
    render_video(parse_args())
