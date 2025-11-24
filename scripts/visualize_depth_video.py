"""Convert a depth sequence (stacked .npy or folder of .npy frames) to an MP4 preview."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import imageio.v2 as imageio
from matplotlib import colormaps as mpl_cmaps
import numpy as np
from tqdm import tqdm
import subprocess
import shlex


class DepthSequence:
    """Lightweight loader for either a stacked .npy file or a directory of per-frame .npy files."""

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
                raise ValueError(f"Expected stacked depth of shape (N,H,W) or (N,1,H,W); got {self.stack.shape}")
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
        raise ValueError(f"Depth frame should be 2D; got {arr.shape}")

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
        raise ValueError(f"Expected 2D depth array; got {arr.shape}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        type=Path,
        help="Stacked .npy (shape N,H,W) or directory containing per-frame .npy depth maps.",
    )
    parser.add_argument("out", type=Path, help="Output video file (e.g., depth_preview.mp4).")
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Output video frame rate. If omitted and --video-ref is provided, fps is probed from that video; otherwise defaults to 24.",
    )
    parser.add_argument(
        "--cmap",
        default="inferno",
        help="Matplotlib colormap name (inferno, magma, viridis, etc.).",
    )
    parser.add_argument(
        "--percentiles",
        type=float,
        nargs=2,
        default=(1.0, 99.0),
        metavar=("LOW", "HIGH"),
        help="Percentile range used to normalize depths (default: 1 99).",
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
        help="If the stack file is a raw memmap without .npy header, provide the shape (frames, height, width).",
    )
    parser.add_argument(
        "--raw-dtype",
        choices=["float16", "float32"],
        default="float16",
        help="Datatype for --raw-shape fallback (default float16).",
    )
    return parser.parse_args()


def compute_global_range(seq: DepthSequence, percentiles: Tuple[float, float], sample_step: int) -> Tuple[float, float]:
    lows: List[float] = []
    highs: List[float] = []
    for depth in seq.iterate(step=max(1, sample_step)):
        finite = depth[np.isfinite(depth)]
        if finite.size == 0:
            continue
        lo, hi = np.percentile(finite, percentiles)
        lows.append(float(lo))
        highs.append(float(hi))
    if not lows:
        raise ValueError("No finite depth values found to compute percentiles.")
    vmin = float(np.median(lows))
    vmax = float(np.median(highs))
    if vmin == vmax:
        vmax = vmin + 1e-6
    return vmin, vmax


def depth_to_rgb(depth: np.ndarray, vmin: float, vmax: float, cmap_name: str) -> np.ndarray:
    """Map a depth frame to uint8 RGB using a colormap."""
    cmap = mpl_cmaps.get_cmap(cmap_name)
    norm = np.clip((depth - vmin) / (vmax - vmin), 0.0, 1.0)
    norm = np.nan_to_num(norm, nan=0.0, posinf=0.0, neginf=0.0)
    rgb_float = cmap(norm)[..., :3]  # drop alpha
    return (rgb_float * 255).astype(np.uint8)


def maybe_write_png(frame_rgb: np.ndarray, idx: int, frames_dir: Optional[Path]) -> None:
    if frames_dir is None:
        return
    frames_dir.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(frames_dir / f"frame{idx:06d}.png", frame_rgb)


def render_video(args: argparse.Namespace) -> None:
    seq = DepthSequence(args.source, raw_shape=tuple(args.raw_shape) if args.raw_shape else None, raw_dtype=args.raw_dtype)
    vmin, vmax = compute_global_range(seq, tuple(args.percentiles), args.sample_step)
    print(f"Color normalization range: [{vmin:.4f}, {vmax:.4f}] using cmap='{args.cmap}'")

    fps_value = resolve_fps(args)

    if args.use_ffmpeg:
        run_ffmpeg_render(seq, args, vmin, vmax, fps_value)
    else:
        writer = imageio.get_writer(args.out, fps=fps_value, codec="libx264", quality=8)
        for idx, depth in enumerate(tqdm(seq.iterate(), total=seq.length, desc="Rendering video")):
            rgb = depth_to_rgb(depth, vmin, vmax, args.cmap)
            writer.append_data(rgb)
            maybe_write_png(rgb, idx, args.frames_dir)
        writer.close()
        print(f"Wrote depth preview to: {args.out}")
        if args.frames_dir:
            print(f"Per-frame PNGs stored in: {args.frames_dir}")


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


def run_ffmpeg_render(seq: DepthSequence, args: argparse.Namespace, vmin: float, vmax: float, fps: float) -> None:
    height, width = seq.shape
    cmd = build_ffmpeg_cmd(args, width, height, fps)
    print("Running ffmpeg:", " ".join(shlex.quote(part) for part in cmd))

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None

    try:
        for idx, depth in enumerate(tqdm(seq.iterate(), total=seq.length, desc="Rendering video")):
            rgb = depth_to_rgb(depth, vmin, vmax, args.cmap)
            proc.stdin.write(rgb.astype(np.uint8).tobytes())
            maybe_write_png(rgb, idx, args.frames_dir)
    finally:
        if proc.stdin:
            proc.stdin.close()
            proc.stdin = None
        stderr = proc.stderr.read() if proc.stderr else b""
        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"ffmpeg failed with code {ret}:\n{stderr.decode(errors='ignore')}")
    print(f"Wrote depth preview to: {args.out} via ffmpeg ({args.encoder}, {args.pix_fmt})")
    if args.frames_dir:
        print(f"Per-frame PNGs stored in: {args.frames_dir}")


def resolve_fps(args: argparse.Namespace) -> float:
    if args.fps:
        return float(args.fps)
    if args.video_ref:
        probed = probe_fps(args.ffmpeg_bin, args.video_ref)
        if probed:
            return probed
    # Fallback default
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
    # ffprobe outputs two lines: r_frame_rate and avg_frame_rate as rationals like 30000/1001
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
