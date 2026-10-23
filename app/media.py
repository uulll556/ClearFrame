from __future__ import annotations

import hashlib
import subprocess
from functools import lru_cache
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

from .config import ROOT_DIR, settings


CACHE_DIR = ROOT_DIR / "cache"
THUMB_DIR = CACHE_DIR / "thumbnails"
MASK_DIR = CACHE_DIR / "masks"
DETECTION_DIR = CACHE_DIR / "detection"
WATERMARK_TEXT = "\u65b0\u7247\u573a\u6b63\u7248\u7d20\u6750"
DETECTION_VERSION = "v5"
TEMPLATE_MARGIN = 4
TEMPLATE_RATIOS = (0.24, 0.28, 0.32, 0.36, 0.40, 0.44, 0.48)
COMPLETENESS_BIAS = 0.05
LOGO_RECOVERY_SCORE_RATIO = 0.74


def ffmpeg_executable() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def source_path(relative_path: str) -> Path:
    return Path(settings.source_dir) / relative_path


def get_video_size(path: Path) -> tuple[int, int]:
    reader = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
    try:
        metadata = next(reader)
        width, height = metadata["size"]
        return int(width), int(height)
    finally:
        reader.close()


def get_video_metadata(path: Path) -> dict:
    reader = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
    try:
        return dict(next(reader))
    finally:
        reader.close()


def get_thumbnail(video_id: int, path: Path, second: float = 1.0) -> Path:
    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    signature = hashlib.sha1(f"{path}:{path.stat().st_mtime_ns}".encode()).hexdigest()[:12]
    target = THUMB_DIR / f"{video_id}-{signature}.jpg"
    if target.exists():
        return target
    subprocess.run(
        [
            ffmpeg_executable(), "-hide_banner", "-loglevel", "error",
            "-ss", str(max(second, 0)), "-i", str(path), "-frames:v", "1",
            "-vf", "scale=640:-2", "-q:v", "3", "-y", str(target),
        ],
        check=True,
    )
    return target


def get_detection_frames(video_id: int, path: Path, frame_count: int = 5) -> list[Image.Image]:
    metadata = get_video_metadata(path)
    duration = max(0.1, float(metadata.get("duration") or 0.1))
    signature = hashlib.sha1(
        f"{path}:{path.stat().st_mtime_ns}:{DETECTION_VERSION}".encode()
    ).hexdigest()[:12]
    target_dir = DETECTION_DIR / f"{video_id}-{signature}"
    existing = sorted(target_dir.glob("*.jpg")) if target_dir.exists() else []
    if len(existing) < 3:
        target_dir.mkdir(parents=True, exist_ok=True)
        for stale in existing:
            stale.unlink(missing_ok=True)
        start = min(1.0, duration * 0.08) if duration > 1 else 0.0
        usable_duration = max(0.1, duration - start * 2)
        interval = max(0.2, usable_duration / max(1, frame_count - 1))
        subprocess.run(
            [
                ffmpeg_executable(), "-hide_banner", "-loglevel", "error",
                "-ss", str(start), "-i", str(path), "-t", str(usable_duration),
                "-vf", f"fps=1/{interval},scale=480:-2", "-frames:v", str(frame_count),
                "-q:v", "4", "-y", str(target_dir / "%02d.jpg"),
            ],
            check=True,
        )
        existing = sorted(target_dir.glob("*.jpg"))
    if not existing:
        existing = [get_thumbnail(video_id, path)]
    return [Image.open(frame).convert("RGB") for frame in existing]


def _build_watermark_template(frame_width: int, ratio: float) -> Image.Image:
    font_path = Path("C:/Windows/Fonts/msyhbd.ttc")
    target_width = int(frame_width * ratio)
    font = ImageFont.truetype(str(font_path), max(9, int(target_width / 7.2)))
    box = font.getbbox(WATERMARK_TEXT)
    text_width = box[2] - box[0]
    text_height = box[3] - box[1]
    logo_size = max(6, text_height)
    logo_gap = max(2, round(text_height * 0.22))
    content_height = max(logo_size, text_height)
    template = Image.new(
        "L",
        (
            logo_size + logo_gap + text_width + TEMPLATE_MARGIN * 2,
            content_height + TEMPLATE_MARGIN * 2,
        ),
        0,
    )
    draw = ImageDraw.Draw(template)
    logo_top = TEMPLATE_MARGIN + (content_height - logo_size) // 2
    outline_width = max(1, round(logo_size * 0.10))
    draw.ellipse(
        (
            TEMPLATE_MARGIN,
            logo_top,
            TEMPLATE_MARGIN + logo_size - 1,
            logo_top + logo_size - 1,
        ),
        outline=255,
        width=outline_width,
    )
    logo_font = ImageFont.truetype(str(font_path), max(5, round(logo_size * 0.27)))
    logo_box = logo_font.getbbox("new")
    logo_width = logo_box[2] - logo_box[0]
    logo_height = logo_box[3] - logo_box[1]
    draw.text(
        (
            TEMPLATE_MARGIN + (logo_size - logo_width) / 2 - logo_box[0],
            logo_top + (logo_size - logo_height) / 2 - logo_box[1],
        ),
        "new",
        font=logo_font,
        fill=255,
    )
    text_left = TEMPLATE_MARGIN + logo_size + logo_gap
    text_top = TEMPLATE_MARGIN + (content_height - text_height) // 2
    draw.text(
        (text_left - box[0], text_top - box[1]),
        WATERMARK_TEXT,
        font=font,
        fill=255,
    )
    return template


@lru_cache(maxsize=8)
def _watermark_templates(frame_width: int) -> tuple[tuple[int, int, object], ...]:
    import numpy as np
    from skimage.feature import canny

    font_path = Path("C:/Windows/Fonts/msyhbd.ttc")
    if not font_path.exists():
        return ()
    templates = []
    for ratio in TEMPLATE_RATIOS:
        template = _build_watermark_template(frame_width, ratio)
        templates.append(
            (
                template.width,
                template.height,
                canny(np.asarray(template) / 255.0, sigma=0.7),
            )
        )
    return tuple(templates)


def _candidate_rank(score: float, ratio: float) -> float:
    return score + ratio * COMPLETENESS_BIAS


def _choose_complete_candidate(
    candidates: list[tuple[float, float, float, int, int, int, int]],
    frame_width: int,
    frame_height: int,
) -> tuple[tuple[float, float, float, int, int, int, int] | None, bool]:
    if not candidates:
        return None, False
    best = max(candidates, key=lambda candidate: candidate[0])
    _, best_score, best_ratio, best_x, best_y, best_width, best_height = best
    best_center_x = (best_x + best_width / 2) / frame_width
    best_center_y = (best_y + best_height / 2) / frame_height
    recover_logo_anchor = best_ratio == 0.32 and any(
        candidate[2] >= 0.40
        and candidate[1] >= best_score * LOGO_RECOVERY_SCORE_RATIO
        and abs((candidate[3] + candidate[5] / 2) / frame_width - best_center_x) <= 0.04
        and abs((candidate[4] + candidate[6] / 2) / frame_height - best_center_y) <= 0.03
        for candidate in candidates
    )
    return best, recover_logo_anchor


def _adaptive_mask_region(
    frame_width: int,
    frame_height: int,
    match_x: int,
    match_y: int,
    template_width: int,
    template_height: int,
    recover_logo_anchor: bool = False,
) -> tuple[float, float, float, float]:
    content_left = match_x + TEMPLATE_MARGIN
    content_top = match_y + TEMPLATE_MARGIN
    content_width = max(1, template_width - TEMPLATE_MARGIN * 2)
    content_height = max(1, template_height - TEMPLATE_MARGIN * 2)
    safety_x = max(1, round(content_height * 0.10))
    safety_y = max(1, round(content_height * 0.12))
    if content_height >= 21:
        safety_x = max(safety_x, round(content_height * 0.22))
        safety_y = max(safety_y, round(content_height * 0.22))
    if content_height >= 23:
        safety_x += 2
        safety_y += 1
    if content_height >= 28:
        safety_x = max(safety_x, round(content_height * 0.43))
        safety_y = max(safety_y, round(content_height * 0.36))

    recovery_x = 5 if recover_logo_anchor else 0
    recovery_y = 2 if recover_logo_anchor else 0
    left = max(0, content_left - safety_x - recovery_x)
    if recover_logo_anchor:
        left = max(0, left - round(content_height * 0.75))
    top = max(0, content_top - safety_y - recovery_y)
    right = min(frame_width, content_left + content_width + safety_x + recovery_x)
    bottom = min(
        frame_height,
        content_top + content_height + safety_y + recovery_y,
    )
    return (
        left / frame_width,
        top / frame_height,
        (right - left) / frame_width,
        (bottom - top) / frame_height,
    )


def detect_region_from_images(images: list[Image.Image]) -> dict[str, float | str | bool]:
    import numpy as np
    from skimage.color import rgb2gray
    from skimage.feature import canny, match_template

    if not images:
        raise ValueError("At least one detection frame is required")
    frame_width, frame_height = images[0].size
    normalized = [
        image if image.size == (frame_width, frame_height)
        else image.resize((frame_width, frame_height), Image.Resampling.BILINEAR)
        for image in images
    ]
    persistent_edges = np.mean(
        [
            canny(rgb2gray(np.asarray(image) / 255.0), sigma=1.0)
            for image in normalized
        ],
        axis=0,
    )
    candidates: list[tuple[float, float, float, int, int, int, int]] = []
    for ratio, template_data in zip(TEMPLATE_RATIOS, _watermark_templates(frame_width)):
        template_width, template_height, template_edges = template_data
        result = match_template(persistent_edges, template_edges, pad_input=False)
        y_grid, x_grid = np.indices(result.shape)
        center_x = (x_grid + template_width / 2) / frame_width
        valid = (
            (center_x > 0.38) & (center_x < 0.62)
            & (y_grid / frame_height > 0.08)
            & ((y_grid + template_height) / frame_height < 0.92)
        )
        scored = np.where(
            valid,
            result - 0.08 * np.abs(center_x - 0.5),
            -9.0,
        )
        y, x = np.unravel_index(np.argmax(scored), scored.shape)
        score = float(scored[y, x])
        candidate = (
            _candidate_rank(score, ratio), score, ratio, int(x), int(y),
            template_width, template_height,
        )
        candidates.append(candidate)

    best, recover_logo_anchor = _choose_complete_candidate(
        candidates, frame_width, frame_height
    )

    if best is None or best[1] < 0.065:
        return {
            "x": settings.mask_x, "y": settings.mask_y,
            "width": settings.mask_width, "height": settings.mask_height,
            "score": best[1] if best else 0.0,
            "method": "fallback_default", "review_required": True,
        }
    _, score, _, match_x, match_y, template_width, template_height = best
    x, y, width, height = _adaptive_mask_region(
        frame_width,
        frame_height,
        match_x,
        match_y,
        template_width,
        template_height,
        recover_logo_anchor,
    )
    return {
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "score": score,
        "method": "multi_frame_complete_scale_v7",
        "review_required": score < 0.11,
    }


def detect_watermark_region(video_id: int, path: Path) -> dict[str, float | str | bool]:
    return detect_region_from_images(get_detection_frames(video_id, path))


def create_mask(video: dict, path: Path) -> Path:
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    width = video.get("width")
    height = video.get("height")
    if not width or not height:
        width, height = get_video_size(path)

    region = (
        video.get("mask_x") if video.get("mask_x") is not None else settings.mask_x,
        video.get("mask_y") if video.get("mask_y") is not None else settings.mask_y,
        video.get("mask_width") if video.get("mask_width") is not None else settings.mask_width,
        video.get("mask_height") if video.get("mask_height") is not None else settings.mask_height,
    )
    x, y, region_width, region_height = region
    box = (
        max(0, round(x * width)),
        max(0, round(y * height)),
        min(width - 1, round((x + region_width) * width)),
        min(height - 1, round((y + region_height) * height)),
    )
    radius = max(1, round(min(box[2] - box[0], box[3] - box[1]) * 0.12))
    mask = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask).rounded_rectangle(box, radius=radius, fill=255)
    if settings.mask_feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(settings.mask_feather))
        clip = Image.new("L", (width, height), 0)
        ImageDraw.Draw(clip).rectangle(box, fill=255)
        mask = ImageChops.multiply(mask, clip)
    target = MASK_DIR / f"{video['id']}.png"
    mask.save(target)
    return target
