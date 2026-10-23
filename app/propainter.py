from __future__ import annotations

import asyncio
import os
import re
import shutil
import subprocess
import sys
from collections import deque
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

from .local_inpaint import SAMPLE_FRAMES, _mux_audio, build_inpaint_mask

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROPAINTER_DIR = PROJECT_ROOT / "tools" / "ProPainter"
PROPAINTER_SCRIPT = PROPAINTER_DIR / "inference_propainter.py"
PROPAINTER_WEIGHTS = (
    "ProPainter.pth",
    "raft-things.pth",
    "recurrent_flow_completion.pth",
)

# ProPainter only has to repaint the watermark band, not the whole frame. On the
# band it can run at native resolution, which keeps the repair sharp while using
# a fraction of the memory of a full-frame pass.
BAND_MARGIN_Y = 120
BAND_MARGIN_X = 90
SUBVIDEO_LENGTH = int(os.environ.get("WATERMARK_PROPAINTER_SUBVIDEO", "8"))
NEIGHBOR_LENGTH = int(os.environ.get("WATERMARK_PROPAINTER_NEIGHBOR", "4"))
RAFT_ITER = 10
MASK_DILATION = 2
MAX_BAND_WIDTH = 1920
MAX_BAND_HEIGHT = 600
# Whole-frame fallback caps (WATERMARK_PROPAINTER_BAND=0).
MAX_WIDTH = int(os.environ.get("WATERMARK_PROPAINTER_MAX_WIDTH", "960"))
MAX_HEIGHT = int(os.environ.get("WATERMARK_PROPAINTER_MAX_HEIGHT", "540"))


def _weights_ready() -> bool:
    weights = PROPAINTER_DIR / "weights"
    return PROPAINTER_SCRIPT.is_file() and all(
        (weights / name).is_file() for name in PROPAINTER_WEIGHTS
    )


def _probe(video_path: Path) -> tuple[int, int, float]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Video not readable: {video_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    capture.release()
    if width <= 0 or height <= 0:
        raise ValueError(f"Video size is unknown: {video_path}")
    return width, height, fps if fps > 1 else 24.0


def _gpu_free_mb() -> int:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return int(result.stdout.strip().splitlines()[0])
    except Exception:  # noqa: BLE001 - any failure means "size unknown"
        return 0


def _write_png(path: Path, image: np.ndarray) -> None:
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise ValueError(f"Could not encode mask for {path}")
    buffer.tofile(str(path))


def _band_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, columns = np.where(mask > 0)
    if not rows.size:
        raise ValueError("Watermark mask is empty")
    height, width = mask.shape
    # Align to multiples of 8 so ProPainter's padded size equals the band size
    # and the repair maps back pixel for pixel.
    x0 = max(0, int(columns.min()) - BAND_MARGIN_X) // 8 * 8
    y0 = max(0, int(rows.min()) - BAND_MARGIN_Y) // 8 * 8
    x1 = min(width, -(-(int(columns.max()) + 1 + BAND_MARGIN_X) // 8) * 8)
    y1 = min(height, -(-(int(rows.max()) + 1 + BAND_MARGIN_Y) // 8) * 8)
    return x0, y0, x1, y1


def _write_band(video_path: Path, box: tuple[int, int, int, int], target: Path, fps: float) -> None:
    x0, y0, x1, y1 = box
    width, height = x1 - x0, y1 - y0
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Video not readable: {video_path}")
    writer = imageio_ffmpeg.write_frames(
        str(target),
        (width, height),
        fps=fps,
        codec="libx264",
        pix_fmt_out="yuv420p",
        macro_block_size=1,
        ffmpeg_log_level="error",
        output_params=["-crf", "12", "-preset", "fast"],
    )
    writer.send(None)
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            crop = frame[y0:y1, x0:x1]
            writer.send(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).tobytes())
    finally:
        writer.close()
        capture.release()


def _compose(
    video_path: Path,
    box: tuple[int, int, int, int],
    frames_dir: Path,
    band_mask: np.ndarray,
    target: Path,
    fps: float,
    polish: np.ndarray | None = None,
) -> None:
    """Paste the repaired band back into the original frames."""
    frame_files = sorted(frames_dir.glob("*.png"))
    if not frame_files:
        raise ValueError("ProPainter did not produce output frames")
    x0, y0, x1, y1 = box
    band_width, band_height = x1 - x0, y1 - y0
    alpha = cv2.dilate(
        band_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1
    ).astype(np.float32) / 255.0
    # A narrow feather keeps the seam tight; a wide one reads as a blurry halo.
    alpha = cv2.GaussianBlur(alpha, (0, 0), 1.2)[..., None]
    polish = polish if polish is not None else np.zeros_like(band_mask)
    has_polish = bool(polish.any())

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Video not readable: {video_path}")
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = imageio_ffmpeg.write_frames(
        str(target),
        (frame_width, frame_height),
        fps=fps,
        codec="libx264",
        pix_fmt_out="yuv420p",
        macro_block_size=1,
        ffmpeg_log_level="error",
        output_params=["-crf", "18", "-preset", "medium"],
    )
    writer.send(None)
    try:
        for frame_file in frame_files:
            ok, frame = capture.read()
            if not ok:
                break
            repaired = cv2.imdecode(
                np.fromfile(str(frame_file), dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if repaired.shape[1] != band_width or repaired.shape[0] != band_height:
                repaired = cv2.resize(
                    repaired,
                    (band_width, band_height),
                    interpolation=cv2.INTER_LANCZOS4,
                )
            region = frame[y0:y1, x0:x1]
            blended = repaired.astype(np.float32) * alpha + region.astype(np.float32) * (1 - alpha)
            merged = np.clip(blended, 0, 255).astype(np.uint8)
            if has_polish:
                merged = cv2.inpaint(merged, polish, 1, cv2.INPAINT_TELEA)
            # Restore micro-detail: the model output is resampled and a little
            # smooth, which otherwise reads as a blurred patch.
            detail = cv2.GaussianBlur(merged, (0, 0), 1.3)
            sharpened = cv2.addWeighted(merged, 2.0, detail, -1.0, 0)
            merged = np.clip(
                sharpened.astype(np.float32) * alpha
                + merged.astype(np.float32) * (1 - alpha),
                0,
                255,
            ).astype(np.uint8)
            frame[y0:y1, x0:x1] = merged
            writer.send(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).tobytes())
    finally:
        writer.close()
        capture.release()


async def _run_propainter(
    video_path: Path,
    mask_path: Path,
    output_dir: Path,
    scale: float,
    fps: float,
    progress,
) -> None:
    if not _weights_ready():
        raise ValueError(
            "ProPainter is not installed. Expected the source tree in "
            f"{PROPAINTER_DIR} and model weights in {PROPAINTER_DIR / 'weights'}. "
            "See README (ProPainter provider) for the download commands."
        )

    environment = os.environ.copy()
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    command = [
        sys.executable, "-u", str(PROPAINTER_SCRIPT),
        "--video", str(video_path),
        "--mask", str(mask_path),
        "--output", str(output_dir),
        "--fp16",
        "--resize_ratio", f"{scale:.4f}",
        "--subvideo_length", str(SUBVIDEO_LENGTH),
        "--neighbor_length", str(NEIGHBOR_LENGTH),
        "--raft_iter", str(RAFT_ITER),
        "--mask_dilation", str(MASK_DILATION),
        "--save_frames",
        "--save_fps", str(int(round(fps))),
    ]
    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=str(PROPAINTER_DIR),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=environment,
    )
    tail: deque[str] = deque(maxlen=30)
    buffer = b""
    last_percent = -1
    try:
        assert process.stdout is not None
        while True:
            chunk = await process.stdout.read(8192)
            if not chunk:
                break
            buffer += chunk
            while True:
                positions = [pos for pos in (buffer.find(b"\r"), buffer.find(b"\n")) if pos >= 0]
                if not positions:
                    break
                cut = min(positions)
                line, buffer = buffer[:cut], buffer[cut + 1:]
                text = line.decode("utf-8", "replace").strip()
                if not text:
                    continue
                tail.append(text)
                match = re.search(r"(\d+)%", text)
                if match:
                    percent = int(match.group(1))
                    if percent != last_percent:
                        last_percent = percent
                        await progress(0.12 + 0.78 * percent / 100.0, None)
        returncode = await process.wait()
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
    if returncode != 0:
        detail = " | ".join(list(tail)[-6:])
        if "out of memory" in detail.lower() or "CUDA error" in detail:
            detail = f"{detail} (the GPU has too little free memory; close GPU apps and retry)"
        raise ValueError(f"ProPainter failed ({returncode}): {detail}")


def _band_scale(band_width: int, band_height: int) -> float:
    """Prefer a native-resolution pass; only shrink when memory is tight."""
    area = band_width * band_height
    native = 1.0 if area <= 600_000 else min(1.0, (600_000 / area) ** 0.5)
    if _gpu_free_mb() < 3500:
        native *= 0.8
    return min(native, MAX_BAND_WIDTH / band_width, MAX_BAND_HEIGHT / band_height)


async def remove_watermark(
    video_path: Path,
    region_mask_path: Path,
    output_path: Path,
    progress,
    grow: int = 2,
) -> None:
    if not PROPAINTER_SCRIPT.is_file():
        raise ValueError(
            f"ProPainter source not found at {PROPAINTER_DIR}; run the setup steps in README."
        )
    _, _, fps = await asyncio.to_thread(_probe, video_path)

    work_dir = output_path.parent / f".{output_path.stem}.propainter"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        mask = await asyncio.to_thread(
            build_inpaint_mask, video_path, region_mask_path, max(0, int(grow))
        )
        full_mask = await asyncio.to_thread(
            build_inpaint_mask,
            video_path,
            region_mask_path,
            max(0, int(grow)),
            SAMPLE_FRAMES,
            "opening",
        )
        polish = cv2.bitwise_and(full_mask, cv2.bitwise_not(mask))
        width, height = mask.shape[1], mask.shape[0]
        band_setting = os.environ.get("WATERMARK_PROPAINTER_BAND")
        # Native-resolution band passes are sharp but lose scene context, which
        # can make a thin high-contrast object (a blade) hallucinate on 1080p
        # sources. Use whole frames there, native bands for smaller clips.
        use_band = (height <= 800) if band_setting is None else band_setting != "0"
        if not use_band:
            # Whole-frame pass: less sharp, but keeps full scene context.
            box = (0, 0, width, height)
            band_mask = mask
            band_polish = polish
            process_video = video_path
            mask_path = work_dir / "full_mask.png"
            await asyncio.to_thread(_write_png, mask_path, band_mask)
            scale = min(1.0, MAX_WIDTH / width, MAX_HEIGHT / height)
        else:
            box = await asyncio.to_thread(_band_box, mask)
            x0, y0, x1, y1 = box
            band_mask = mask[y0:y1, x0:x1]
            band_polish = polish[y0:y1, x0:x1]
            process_video = work_dir / f"{output_path.stem}_band.mp4"
            mask_path = work_dir / "band_mask.png"
            await asyncio.to_thread(_write_band, video_path, box, process_video, fps)
            await asyncio.to_thread(_write_png, mask_path, band_mask)
            scale = _band_scale(x1 - x0, y1 - y0)
        await progress(0.08, None)

        frames_root = work_dir / "frames_out"
        await _run_propainter(process_video, mask_path, frames_root, scale, fps, progress)

        frames_dir = frames_root / process_video.stem / "frames"
        temporary = work_dir / "video-only.mp4"
        await progress(0.92, None)
        await asyncio.to_thread(
            _compose, video_path, box, frames_dir, band_mask, temporary, fps, band_polish
        )
        await asyncio.to_thread(_mux_audio, temporary, video_path, output_path)
        await progress(1.0, None)
    finally:
        if os.environ.get("WATERMARK_PROPAINTER_KEEP") != "1":
            shutil.rmtree(work_dir, ignore_errors=True)
