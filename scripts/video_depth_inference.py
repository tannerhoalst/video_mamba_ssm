"""Run UniDepthV2 on every frame of a video.

Usage
-----
python video_depth_inference.py \\
  --video /path/to/video.mp4 \\
  --output-dir /mnt/vrdata/depth_maps/unidepth/my_run \\
  --model-id lpiccinelli/unidepth-v2-vitb14 \\
  --stack-path /mnt/vrdata/depth_maps/unidepth/my_run_stack.npy \\
  --batch-size 4 --stride 1
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import av
from tqdm import tqdm
import pandas as pd

from unidepth.models import UniDepthV2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True, help="Input video file (e.g., .mp4).")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to store per-frame .npy depth maps (one file per kept frame).",
    )
    parser.add_argument(
        "--model-id",
        default="lpiccinelli/unidepth-v2-vitb14",
        help="Hugging Face repo ID or local checkpoint passed to UniDepthV2.from_pretrained (default: vitb14).",
    )
    parser.add_argument("--device", default=None, help="Torch device string (default: cuda if available else cpu).")
    parser.add_argument("--batch-size", type=int, default=2, help="Frames per forward pass.")
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Keep every Nth frame (stride=1 keeps all frames; stride=5 keeps every 5th frame).",
    )
    parser.add_argument(
        "--stack-dtype",
        choices=["float16", "float32"],
        default="float16",
        help="Datatype used for the stacked array (float16 cuts disk usage roughly in half).",
    )
    parser.add_argument(
        "--save-uncertainty",
        action="store_true",
        help="If set, save UniDepth 'confidence' output per frame and as a stacked .npy in the output directory.",
    )
    parser.add_argument(
        "--uncert-stack-dtype",
        choices=["float16", "float32"],
        default="float16",
        help="Datatype used for the stacked uncertainty array.",
    )
    parser.add_argument(
        "--skip-individual",
        action="store_true",
        help="Skip writing per-frame .npy files; only write the stacked file if --stack-path is provided.",
    )
    parser.add_argument(
        "--decoder",
        choices=["pyav", "opencv"],
        default="pyav",
        help="Video decoding backend. 'pyav' uses your FFmpeg build (recommended if hardware accel is enabled).",
    )
    parser.add_argument(
        "--hwaccel",
        help="FFmpeg hwaccel string (e.g., cuda, cuvid, vaapi, qsv). Only used with --decoder pyav.",
    )
    parser.add_argument(
        "--hwaccel-output-format",
        help="FFmpeg hwaccel_output_format (e.g., cuda). Only used with --decoder pyav.",
    )
    parser.add_argument(
        "--ffmpeg-threads",
        type=int,
        help="Override FFmpeg thread count. Only used with --decoder pyav.",
    )
    parser.add_argument("--fx", type=float, help="Optional pinhole fx (pixels) for all frames.")
    parser.add_argument("--fy", type=float, help="Optional pinhole fy (pixels) for all frames.")
    parser.add_argument("--cx", type=float, help="Optional pinhole cx (pixels) for all frames.")
    parser.add_argument("--cy", type=float, help="Optional pinhole cy (pixels) for all frames.")
    parser.add_argument(
        "--intrinsics-csv",
        type=Path,
        help="Optional metadata_camera_parameters.csv; if provided, intrinsics are read from this file.",
    )
    parser.add_argument(
        "--camera-name",
        default="cam_00",
        help="Camera name to select from --intrinsics-csv (default: cam_00). Ignored if fx/fy/cx/cy are given.",
    )
    return parser.parse_args()


def load_model(model_id: str, device_str: Optional[str]) -> tuple[UniDepthV2, torch.device]:
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = UniDepthV2.from_pretrained(model_id)
    model = model.to(device).eval()
    return model, device


def parse_intrinsics_csv(csv_path: Path, camera_name: str) -> np.ndarray:
    """
    Parse metadata_camera_parameters.csv to get fx, fy, cx, cy.
    Supports either a single 'M_proj' column (16 space-separated floats) or M_proj_00..33 columns.
    Uses the first row matching camera_name; if camera_name not found, falls back to the first row.
    """
    df = pd.read_csv(csv_path)
    if "camera_name" in df.columns:
        df_cam = df[df["camera_name"] == camera_name]
        if not df_cam.empty:
            df = df_cam
    row = df.iloc[0]

    W = float(row["settings_output_img_width"])
    H = float(row["settings_output_img_height"])

    if "M_proj" in row:
        vals = [float(x) for x in str(row["M_proj"]).strip().split()]
        if len(vals) != 16:
            raise ValueError(f"M_proj in {csv_path} is not 16 values")
    else:
        vals = []
        for i in range(4):
            for j in range(4):
                key = f"M_proj_{i}{j}"
                if key not in row:
                    raise ValueError(f"Missing {key} in {csv_path}")
                vals.append(float(row[key]))
    m00, m01, m02, m03, m10, m11, m12, m13, m20, m21, m22, m23, m30, m31, m32, m33 = vals
    fx = m00 * W / 2.0
    fy = m11 * H / 2.0
    cx = (m02 + 1.0) * W / 2.0
    cy = (m12 + 1.0) * H / 2.0
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    return K


def pyav_stream(path: Path, stride: int, hwaccel: Optional[str], hwaccel_output: Optional[str], threads: Optional[int]) -> tuple[Iterable[Tuple[int, np.ndarray]], Optional[int]]:
    """Yield (frame_idx, frame_rgb24 ndarray) and return expected frame count if known."""
    opts: Dict[str, str] = {}
    if hwaccel:
        opts["hwaccel"] = hwaccel
    if hwaccel_output:
        opts["hwaccel_output_format"] = hwaccel_output
    if threads is not None:
        opts["threads"] = str(threads)

    container = av.open(str(path), options=opts or None)
    video_stream = container.streams.video[0]
    total_frames = video_stream.frames if video_stream.frames > 0 else None

    def generator():
        for i, frame in enumerate(container.decode(video_stream)):
            if i % stride != 0:
                continue
            # Convert to RGB24 numpy array (H, W, 3)
            rgb = frame.to_ndarray(format="rgb24")
            yield i, rgb

    expected = math.ceil(total_frames / stride) if total_frames else None
    return generator(), expected


def opencv_stream(path: Path, stride: int) -> tuple[Iterable[Tuple[int, np.ndarray]], Optional[int]]:
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    expected = math.ceil(total / stride) if total > 0 else None

    def generator():
        idx = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                yield idx, rgb
            idx += 1
        cap.release()

    return generator(), expected


def make_memmap(
    stack_path: Path, dtype: np.dtype, frames: int, height: int, width: int
) -> np.memmap:
    stack_path.parent.mkdir(parents=True, exist_ok=True)
    # open_memmap writes a valid .npy header so downstream np.load works.
    return np.lib.format.open_memmap(
        stack_path, mode="w+", dtype=dtype, shape=(frames, height, width)
    )


def to_tensor(frame_rgb: np.ndarray) -> torch.Tensor:
    """Convert HxWx3 RGB uint8 to CxHxW tensor."""
    tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).contiguous()
    return tensor


def run_inference(args: argparse.Namespace) -> None:
    model, device = load_model(args.model_id, args.device)

    if args.decoder == "pyav":
        frame_iter, expected_frames = pyav_stream(args.video, args.stride, args.hwaccel, args.hwaccel_output_format, args.ffmpeg_threads)
    else:
        frame_iter, expected_frames = opencv_stream(args.video, args.stride)

    stack_dtype = np.float16 if args.stack_dtype == "float16" else np.float32
    stack_mm: Optional[np.memmap] = None
    uncert_mm: Optional[np.memmap] = None
    stack_index = 0

    # Optional fixed intrinsics
    K_fixed: Optional[torch.Tensor] = None
    if all(v is not None for v in (args.fx, args.fy, args.cx, args.cy)):
        K_np = np.array(
            [[args.fx, 0.0, args.cx], [0.0, args.fy, args.cy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        K_fixed = torch.from_numpy(K_np).to(device)
    elif any(v is not None for v in (args.fx, args.fy, args.cx, args.cy)):
        raise ValueError("If specifying intrinsics, provide all of fx, fy, cx, cy.")
    elif args.intrinsics_csv:
        K_np = parse_intrinsics_csv(args.intrinsics_csv, args.camera_name)
        K_fixed = torch.from_numpy(K_np).to(device)

    # Resolve stack paths automatically in the output directory.
    video_stem = args.video.stem
    depth_stack_path = args.output_dir / f"{video_stem}_depth_stack.npy"
    uncert_stack_path = args.output_dir / f"{video_stem}_uncert_stack.npy" if args.save_uncertainty else None

    args.output_dir.mkdir(parents=True, exist_ok=True)

    batch: List[torch.Tensor] = []
    batch_indices: List[int] = []

    pbar_total = expected_frames if expected_frames is not None else None
    pbar = tqdm(total=pbar_total, desc="Processing frames")

    for frame_id, frame_rgb in frame_iter:
        batch.append(to_tensor(frame_rgb))
        batch_indices.append(frame_id)

        if len(batch) >= args.batch_size:
            stack_mm, uncert_mm, stack_index = process_batch(
                batch,
                batch_indices,
                model,
                device,
                args,
                K_fixed,
                stack_mm,
                uncert_mm,
                stack_dtype,
                stack_index,
                expected_frames,
            args.uncert_stack_dtype,
            args.save_uncertainty,
            depth_stack_path,
            uncert_stack_path,
        )
            pbar.update(len(batch_indices))
            batch.clear()
            batch_indices.clear()

    if batch:
        stack_mm, uncert_mm, stack_index = process_batch(
            batch,
            batch_indices,
            model,
            device,
            args,
            K_fixed,
            stack_mm,
            uncert_mm,
            stack_dtype,
            stack_index,
            expected_frames,
            args.uncert_stack_dtype,
            args.save_uncertainty,
            depth_stack_path,
            uncert_stack_path,
        )
        pbar.update(len(batch_indices))

    pbar.close()

    if stack_mm is not None:
        stack_mm.flush()
        filled = stack_index
        total = stack_mm.shape[0]
        note = ""
        if filled < total:
            note = f" (filled {filled} / {total} frames; remaining entries left at zero)"
        print(f"Stacked depth saved to: {depth_stack_path} with shape {stack_mm.shape} and dtype {stack_mm.dtype}{note}")
    if uncert_mm is not None:
        uncert_mm.flush()
        filled = stack_index
        total = uncert_mm.shape[0]
        note = ""
        if filled < total:
            note = f" (filled {filled} / {total} frames; remaining entries left at zero)"
        print(f"Stacked uncertainty saved to: {uncert_stack_path} with shape {uncert_mm.shape} and dtype {uncert_mm.dtype}{note}")


def process_batch(
    frames: List[torch.Tensor],
    indices: List[int],
    model: UniDepthV2,
    device: torch.device,
    args: argparse.Namespace,
    K_fixed: Optional[torch.Tensor],
    stack_mm: Optional[np.memmap],
    uncert_mm: Optional[np.memmap],
    stack_dtype: np.dtype,
    stack_index: int,
    expected_frames: Optional[int],
    uncert_stack_dtype: str,
    save_uncertainty: bool,
    depth_stack_path: Path,
    uncert_stack_path: Optional[Path],
) -> tuple[Optional[np.memmap], Optional[np.memmap], int]:
    batch = torch.stack(frames, dim=0).to(device)
    camera = None
    if K_fixed is not None:
        camera = K_fixed.expand(batch.shape[0], 3, 3)
    with torch.no_grad():
        preds = model.infer(batch, camera=camera, normalize=True)
    depths = preds["depth"].cpu()  # (B, 1, H, W)
    confidences = preds.get("confidence")
    if confidences is not None:
        confidences = confidences.cpu()

    for idx, (depth_tensor, frame_id) in enumerate(zip(depths, indices)):
        depth_np = depth_tensor.squeeze(0).numpy()
        conf_np = confidences[idx].squeeze(0).numpy() if confidences is not None else None

        if depth_stack_path and stack_mm is None:
            # Lazy create memmap after first depth so we know H and W.
            if args.stride <= 0:
                raise ValueError("Stride must be >= 1.")
            height, width = depth_np.shape
            if depth_stack_path.exists():
                depth_stack_path.unlink()
            if expected_frames is None:
                raise ValueError(
                    "Video frame count unavailable; cannot preallocate stacked .npy. Try without --stack-path."
                )
            stack_mm = make_memmap(depth_stack_path, stack_dtype, expected_frames, height, width)
        if save_uncertainty and uncert_stack_path and conf_np is not None and uncert_mm is None:
            if uncert_stack_path.exists():
                uncert_stack_path.unlink()
            if expected_frames is None:
                raise ValueError(
                    "Video frame count unavailable; cannot preallocate uncertainty stacked .npy. Provide expected frames."
                )
            uncert_dtype_np = np.float16 if uncert_stack_dtype == "float16" else np.float32
            uncert_mm = make_memmap(uncert_stack_path, uncert_dtype_np, expected_frames, conf_np.shape[0], conf_np.shape[1])

        wrote_stack = False
        if stack_mm is not None:
            stack_mm[stack_index] = depth_np.astype(stack_dtype, copy=False)
            wrote_stack = True
        if save_uncertainty and conf_np is not None and uncert_mm is not None:
            uncert_dtype_np = np.float16 if uncert_stack_dtype == "float16" else np.float32
            uncert_mm[stack_index] = conf_np.astype(uncert_dtype_np, copy=False)
            wrote_stack = True
        if wrote_stack:
            stack_index += 1

        if not args.skip_individual:
            out_name = f"{args.video.stem}_frame{frame_id:06d}.npy"
            np.save(args.output_dir / out_name, depth_np)
        if save_uncertainty and conf_np is not None and not args.skip_individual:
            out_name_conf = f"{args.video.stem}_frame{frame_id:06d}_uncert.npy"
            np.save(args.output_dir / out_name_conf, conf_np)

    return stack_mm, uncert_mm, stack_index


if __name__ == "__main__":
    run_inference(parse_args())
