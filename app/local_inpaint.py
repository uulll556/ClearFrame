from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from patchmatch import patch_match

from .media import ffmpeg_executable


ROI_PADDING = 40
CROP_PADDING = 110
SAMPLE_FRAMES = 48
RESIDUAL_THRESHOLD = 16
STRUCTURE_BRIGHT_LEVEL = 200
STRUCTURE_BRIGHT_RATIO = 0.005
PATCH_SIZE = 5


def _imread_gray(path: Path) -> np.ndarray | None:
    # cv2.imread cannot open non-ASCII paths on Windows, so decode the bytes.
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


def _mask_bbox(mask_path: Path) -> tuple[int, int, int, int]:
    mask = _imread_gray(mask_path)
    if mask is None:
        raise FileNotFoundError(f"Mask not readable: {mask_path}")
    points = cv2.findNonZero((mask > 127).astype(np.uint8))
    if points is None:
        raise ValueError("Mask is empty")
    x, y, width, height = cv2.boundingRect(points)
    return x, y, x + width, y + height


def _core_from_residual(residual: np.ndarray) -> np.ndarray:
    raw = (residual > RESIDUAL_THRESHOLD).astype(np.uint8) * 255
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    core = cv2.morphologyEx(raw, cv2.MORPH_OPEN, kernel3)
    return cv2.morphologyEx(core, cv2.MORPH_CLOSE, kernel9)


def build_inpaint_mask(
    video_path: Path,
    region_mask_path: Path,
    grow: int = 2,
    sample_limit: int = SAMPLE_FRAMES,
    mode: str | None = None,
) -> np.ndarray:
    """Build a glyph-shaped repair mask from the persistent watermark signal.

    The static overlay is isolated from the moving background with a temporal
    median of the local contrast residual, so the returned mask follows the
    lettering instead of blanking the whole rectangle. Two background estimates
    are sampled: a median blur (stable, conservative) and a wide morphological
    opening (recovers strokes over dense text). Conservative masks are used when
    the region contains bright, thin structures such as a blade.
    """
    x0, y0, x1, y1 = _mask_bbox(region_mask_path)
    roi_x0 = max(0, x0 - ROI_PADDING)
    roi_y0 = max(0, y0 - ROI_PADDING)
    roi_x1 = x1 + ROI_PADDING
    roi_y1 = y1 + ROI_PADDING

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Video not readable: {video_path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    full_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    full_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    roi_x1 = min(full_width, roi_x1)
    roi_y1 = min(full_height, roi_y1)
    if roi_x1 <= roi_x0 or roi_y1 <= roi_y0:
        capture.release()
        raise ValueError("Invalid watermark region")

    total = max(1, min(sample_limit, frame_count))
    indices = [round(i * (frame_count - 1) / (total - 1)) for i in range(total)]
    indices = sorted(set(max(0, index) for index in indices))

    residuals_median: list[np.ndarray] = []
    residuals_open: list[np.ndarray] = []
    samples: list[np.ndarray] = []
    opening_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        patch = gray[roi_y0:roi_y1, roi_x0:roi_x1]
        samples.append(patch.astype(np.float32))
        residuals_median.append(
            patch.astype(np.float32) - cv2.medianBlur(patch, 31).astype(np.float32)
        )
        residuals_open.append(
            patch.astype(np.float32)
            - cv2.morphologyEx(patch, cv2.MORPH_OPEN, opening_kernel).astype(np.float32)
        )
    capture.release()
    if not residuals_median:
        raise ValueError("Unable to sample frames for mask detection")

    median_frame = np.median(np.stack(samples), axis=0)
    bright_ratio = float((median_frame > STRUCTURE_BRIGHT_LEVEL).mean())
    use_median = mode == "median" or (
        mode != "opening" and bright_ratio > STRUCTURE_BRIGHT_RATIO
    )
    if use_median:
        residual = np.median(np.stack(residuals_median), axis=0)
    else:
        residual = np.median(np.stack(residuals_open), axis=0)
    core = _core_from_residual(residual)

    rows = (core > 0).sum(axis=1)
    if rows.max() < 8:
        mask = np.zeros((full_height, full_width), np.uint8)
        mask[roi_y0:roi_y1, roi_x0:roi_x1] = 255
        return mask

    threshold = max(8, float(rows.max()) * 0.12)
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for y, value in enumerate(rows):
        if value >= threshold and start is None:
            start = y
        elif value < threshold and start is not None:
            runs.append((start, y - 1))
            start = None
    if start is not None:
        runs.append((start, len(rows) - 1))
    if not runs:
        raise ValueError("Watermark text band not found")
    top, bottom = max(runs, key=lambda run: rows[run[0]:run[1] + 1].sum())

    band = core[top:bottom + 1]
    columns = (band > 0).sum(axis=0)
    column_threshold = max(3, (bottom - top + 1) * 0.10)
    dense = np.where(columns >= column_threshold)[0]
    if dense.size == 0:
        raise ValueError("Watermark glyph columns not found")
    left, right = int(dense.min()), int(dense.max())
    height = bottom - top + 1
    center_y = (top + bottom) // 2

    glyphs = np.zeros_like(core)
    glyphs[max(0, top - 2):bottom + 3, :] = (
        core[max(0, top - 2):bottom + 3, :] > 0
    ).astype(np.uint8) * 255
    glyphs[:, :left] = 0
    glyphs[:, right + 4:] = 0

    bubble_width = int(height * 1.3)
    cv2.ellipse(
        glyphs,
        (left + bubble_width // 2, center_y),
        (max(6, bubble_width // 2), max(6, int(height * 0.55))),
        0, 0, 360, 255, -1,
    )

    contours, _ = cv2.findContours(glyphs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(glyphs)
    cv2.drawContours(filled, contours, -1, 255, -1)
    if grow > 0:
        filled = cv2.dilate(
            filled,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=grow,
        )

    mask = np.zeros((full_height, full_width), np.uint8)
    mask[roi_y0:roi_y1, roi_x0:roi_x1] = filled
    return mask


def _inpaint(frame: np.ndarray, mask: np.ndarray, box: tuple[int, int, int, int], patch_size: int) -> np.ndarray:
    x0, y0, x1, y1 = box
    crop = np.ascontiguousarray(frame[y0:y1, x0:x1])
    hole = np.ascontiguousarray(mask[y0:y1, x0:x1])
    if not hole.any():
        return frame
    repaired = patch_match.inpaint(crop, hole, patch_size=patch_size)
    result = frame.copy()
    result[y0:y1, x0:x1] = repaired
    return result


def _mux_audio(video_only: Path, source: Path, target: Path) -> None:
    command = [
        ffmpeg_executable(), "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(video_only), "-i", str(source),
        "-map", "0:v:0", "-map", "1:a:0?",
        "-c", "copy", "-shortest", "-movflags", "+faststart", str(target),
    ]
    subprocess.run(command, check=True)


async def remove_watermark(
    video_path: Path,
    region_mask_path: Path,
    output_path: Path,
    progress,
    grow: int = 2,
    patch_size: int = PATCH_SIZE,
) -> None:
    mask = await asyncio.to_thread(build_inpaint_mask, video_path, region_mask_path, grow)
    points = cv2.findNonZero(mask)
    if points is None:
        raise ValueError("Repair mask is empty")
    x, y, width, height = cv2.boundingRect(points)
    full_height, full_width = mask.shape
    box = (
        max(0, x - CROP_PADDING),
        max(0, y - CROP_PADDING),
        min(full_width, x + width + CROP_PADDING),
        min(full_height, y + height + CROP_PADDING),
    )

    reader = imageio_ffmpeg.read_frames(str(video_path), pix_fmt="rgb24")
    metadata = next(reader)
    frame_width, frame_height = metadata["size"]
    fps = float(metadata.get("fps") or 25.0)
    duration = float(metadata.get("duration") or 0.0)
    expected = max(1, int(round(duration * fps))) if duration else 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.stem + ".video-only.mp4")
    writer = imageio_ffmpeg.write_frames(
        str(temporary),
        (frame_width, frame_height),
        fps=fps,
        codec="libx264",
        pix_fmt_out="yuv420p",
        macro_block_size=1,
        ffmpeg_log_level="error",
        output_params=["-crf", "18", "-preset", "medium"],
    )
    writer.send(None)
    processed = 0
    try:
        await progress(0.05, None)
        for raw in reader:
            frame = np.frombuffer(raw, np.uint8).reshape(frame_height, frame_width, 3)
            repaired = await asyncio.to_thread(_inpaint, frame, mask, box, patch_size)
            writer.send(repaired.tobytes())
            processed += 1
            if processed % 5 == 0 or processed == expected:
                ratio = processed / expected if expected else 0.0
                await progress(min(0.9, 0.05 + ratio * 0.85), None)
    finally:
        writer.close()
        reader.close()

    await asyncio.to_thread(_mux_audio, temporary, video_path, output_path)
    temporary.unlink(missing_ok=True)
    await progress(1.0, None)
